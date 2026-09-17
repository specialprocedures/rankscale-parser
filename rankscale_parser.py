import argparse
import hashlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

import tldextract
import weightipy as wp

QUERY_COLS = ["brand_reference", "topic_name", "tags", "query"]

WEIGHT_COLS = ["ai_engine"]

AI_COLS = [
    "ai_engine",
    "ai_overview",
    "chatgpt_websearch",
    "chatgpt_model",
    "chatgpt_shopping",
]

RESPONSE_COLS = [
    "timestamp",
    "result_text",
    "web_search_queries",
    "related_prompts",
]

CITE_COLS = ["brand_citations", "other_citations"]


def load_rankscale_csv(path):
    # Load in data and drop empty cols
    df = pd.read_csv(
        path,
        encoding="utf-16",
        sep="\t",
        converters={"result_text": lambda x: x.replace("\\n", "\n")},
    )

    df.drop(columns=[i for i in df.columns if "Unnamed" in i], inplace=True)

    # Set timestamp to datetime, raise error if it fails
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    # Generate id columns for queries and responses
    query_id = df.apply(lambda row: quick_hash(row, QUERY_COLS + AI_COLS), axis=1)
    df.insert(0, "query_id", query_id)

    response_id = df.apply(
        lambda row: quick_hash(row, RESPONSE_COLS + QUERY_COLS), axis=1
    )
    df.insert(1, "response_id", response_id)

    return df


# Create id column by hashing the specified columns
def quick_hash(row, hash_cols):
    """Generates hash from pandas columns, used for indexing

    Args:
        row (pd.Series): A row from the dataframe.
        hash_cols (list): List of column names to include in the hash.

    Returns:
        str: The SHA-256 hash of the concatenated column values.
    """
    hash_input = "".join(str(row[col]) for col in hash_cols)
    return hashlib.sha256(hash_input.encode()).hexdigest()


def parse_citations(response_df: pd.DataFrame) -> list:
    """Parses cited urls from dataframe into a list of dicts

    Args:
        response_df (pd.DataFrame): df from rankscale export

    Returns:
        list: List of dicts in the format:
        {
            "response_id": str, # id of the response this citation was found in
            "url": str, # the cited url
            "domain": str, # the top domain under public suffix (e.g. "example.com")
            "fqdn": str, # the fully qualified domain name (e.g. "sub.example.com")
        }
    """
    # Get all the columns with citations in
    cite_cols = [i for i in response_df.columns if "citation" in i]

    # Create list to return
    citations = []

    # Iterate over df and each of the cite_cols, extract urls and domains
    for idx, row in response_df.iterrows():
        for col in cite_cols:

            # Skip empties
            if pd.notna(row[col]) and row[col].strip() != "-":

                # Get all urls in the cell
                cell_urls = [url.strip() for url in row[col].split(",")]

                # Pop them all out as separate citations, with the response_id
                for url in cell_urls:
                    parsed_url = tldextract.extract(url)
                    if parsed_url.domain and parsed_url.suffix:
                        d = {
                            "response_id": row["response_id"],
                            "url": url,
                            "domain": parsed_url.top_domain_under_public_suffix.replace(
                                "www.", ""
                            ),
                            "fqdn": parsed_url.fqdn.replace("www.", ""),
                        }
                        citations.append(d)

    return citations


def citations_to_df(citations: list) -> pd.DataFrame:
    """Converts citations from `parse_citations` to pandas dataframe, adding id cols

    Args:
        citations (list): List of dicts in the format:
        {
            "response_id": str, # id of the response this citation was found in
            "url": str, # the cited url
            "domain": str, # the top domain under public suffix (e.g. "example.com")
            "fqdn": str, # the fully qualified domain name (e.g. "sub.example.com")
        }

    Returns:
        pd.DataFrame: DataFrame with citations and their metadata, with
        citation_id and domain_id columns
    """

    df = pd.DataFrame(citations).drop_duplicates(["response_id", "url"])

    # Hash url and response_id to create citation_id
    citation_ids = df.apply(
        lambda row: quick_hash(row, hash_cols=["url", "response_id"]), axis=1
    )
    df.insert(0, "citation_id", citation_ids)

    # Hash domain to create domain_id, linking to the domains table
    domain_ids = df.apply(lambda row: quick_hash(row, hash_cols=["domain"]), axis=1)
    df.insert(2, "domain_id", domain_ids)

    return df


def make_citations_df(response_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the citations dataframe

    Args:
        response_df (pd.DataFrame): DataFrame containing the responses with citations to be processed.

    Returns:
        pd.DataFrame: DataFrame containing the processed citations with weights and metadata.
    """

    # Build the citations df
    citation_list = parse_citations(response_df)
    citations = citations_to_df(citation_list).merge(
        response_df[["response_id"] + WEIGHT_COLS], on="response_id"
    )

    weight_targets = {}
    for col in WEIGHT_COLS:
        col_pcts = response_df[col].value_counts(normalize=True).to_dict()
        weight_targets[col] = col_pcts

    scheme = wp.scheme_from_dict(weight_targets)
    citations_weighted = wp.weight_dataframe(citations, scheme, weight_column="weights")

    return citations_weighted.drop(columns=WEIGHT_COLS)


def make_domains_table(
    citations: pd.DataFrame, responses: pd.DataFrame
) -> pd.DataFrame:
    by_domain = citations.groupby(["domain_id", "domain"])

    domains = pd.DataFrame(
        {
            "at_least_once": by_domain["response_id"].nunique(),
            "proportion": by_domain.size() / len(citations),
            "proportion_weighted": by_domain["weights"].sum()
            / citations["weights"].sum(),
        }
    )
    domains.insert(1, "at_least_once_pct", domains["at_least_once"] / len(responses))

    return domains.sort_values("proportion_weighted", ascending=False).reset_index()


def parse_args(argv=None):
    try:
        pkg_version = version("rankscale-parser")
    except PackageNotFoundError:
        pkg_version = "unknown"

    parser = argparse.ArgumentParser(
        prog="rankscale-parser",
        description="Parse a Rankscale export into queries, responses, citations and domains tables.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Rankscale export to parse (UTF-16, tab-separated)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=Path("."),
        type=Path,
        help="Directory to write output CSVs to (default: current directory)",
    )
    parser.add_argument("--version", action="version", version=pkg_version)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.input.is_file():
        sys.exit(f"rankscale-parser: input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_rankscale_csv(args.input)
    citations = make_citations_df(df)
    domains = make_domains_table(citations, df)

    queries = df[["query_id"] + QUERY_COLS + AI_COLS].drop_duplicates("query_id")
    responses = df[["response_id"] + RESPONSE_COLS]

    # Write to csv
    for write_df, name in [
        (queries, "queries"),
        (responses, "responses"),
        (citations, "citations"),
        (domains, "domains"),
    ]:
        out_path = args.output_dir / f"{name}.csv"
        write_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
