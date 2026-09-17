import json
import hashlib

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
    "result_text",
]

PATH = "redmeat_v2.csv"

CITE_COLS = ["brand_citations", "other_citations"]


def load_rankscale_csv():
    # Load in data and drop empty cols
    df = pd.read_csv(
        PATH,
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

    # Set request_id as index, as this is the unique identifier for each row
    df.set_index("response_id", inplace=True)
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


def parse_urls(query_df: pd.DataFrame) -> list:
    """Parses urls from dataframe into a list of dicts

    Args:
        query_df (pd.DataFrame): df from rankscale export

    Returns:
        list: List of dicts in the format:
        {
            "response_id": str, # id of the response this url was found in
            "query_id": str, # id of the query this url was found in
            "url": str, # the url itself
            "domain": str, # the top domain under public suffix (e.g. "example.com")
            "fqdn": str, # the fully qualified domain name (e.g. "sub.example.com")
        }
    """
    # Get all the columns with urls in
    cite_cols = [i for i in query_df.columns if "citation" in i]

    # Create list to return
    urls = []

    # Iterate over df and each of the cite_cols, extract urls and domains
    for idx, row in query_df.reset_index().iterrows():
        for col in cite_cols:

            # Skip empties
            if pd.notna(row[col]) and row[col].strip() != "-":

                # Get all urls in the cell
                cell_urls = [url.strip() for url in row[col].split(",")]

                # Pop them all out as separate rows in the urls df, with the query_id and source
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
                        urls.append(d)

    return urls


def urls_to_df(urls: list) -> pd.DataFrame:
    """Converts urls from `parse_urls` to pandas dataframe, adding id col

    Args:
        urls (list): List of dicts in the format:
        {
            "response_id": str, # id of the response this url was found in
            "url": str, # the url itself
            "domain": str, # the top domain under public suffix (e.g. "example.com")
            "fqdn": str, # the fully qualified domain name (e.g. "sub.example.com")
        }

    Returns:
        pd.DataFrame: DataFrame with urls and their metadata, indexed by url_id
    """

    df = pd.DataFrame(urls)

    # Hash url and query_id to create url_id
    url_ids = df.apply(
        lambda row: quick_hash(row, hash_cols=["query_id", "url", "response_id"]),
        axis=1,
    )
    df.insert(0, "url_id", url_ids, allow_duplicates=False)
    df.set_index("url_id", inplace=True)

    return df.drop_duplicates(["query_id", "url"]).drop(
        columns=["source_column", "suffix"]
    )


def make_urls_df(response_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the citations dataframe

    Args:
        response_df (pd.DataFrame): DataFrame containing the responses with URLs to be processed.

    Returns:
        pd.DataFrame: DataFrame containing the processed URLs with weights and metadata.
    """

    # Build the urls df
    url_list = parse_urls(response_df)
    urls = urls_to_df(url_list).merge(
        response_df[WEIGHT_COLS],
        right_index=True,
        left_on="response_id",
    )

    weight_targets = {}
    for col in WEIGHT_COLS:
        col_pcts = response_df[col].value_counts(normalize=True).to_dict()
        weight_targets[col] = col_pcts

    scheme = wp.scheme_from_dict(weight_targets)
    urls_weighted = wp.weight_dataframe(urls, scheme, weight_column="weights")

    return urls_weighted.drop(columns=WEIGHT_COLS)


def make_domains_table(urls: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    at_least_once = urls.groupby("domain")["response_id"].nunique()
    at_least_once_pct = at_least_once.div(len(queries))
    prop = urls["domain"].value_counts(normalize=True)
    sov = urls.groupby("domain")["weights"].sum() / urls["weights"].sum()

    domains = pd.concat(
        [at_least_once, at_least_once_pct, prop, sov], axis=1
    ).reset_index()

    domains.columns = [
        "domain",
        "at_least_once",
        "at_least_once_pct",
        "proportion",
        "proportion_weighted",
    ]

    domains.set_index("domain", inplace=True)
    domains["mean"] = domains.mean(axis=1)
    domains.sort_values("mean", ascending=False, inplace=True)

    return domains


def main():
    df = load_rankscale_csv()
    urls = make_urls_df(df)
    domains = make_domains_table(urls, df)

    queries = df[["query_id"] + QUERY_COLS + AI_COLS]
    responses = df[["response_id"] + RESPONSE_COLS]

    # Write to csv
    for write_df, name in zip(
        [queries, responses, urls, domains], ["queries", "responses", "urls", "domains"]
    ):
        write_df.to_csv(f"{name}.csv", index=False, encoding="utf-8")
