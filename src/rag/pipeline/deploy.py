"""Deploy the daily ingestion flow to Prefect."""

from prefect import serve

from rag.pipeline.flows import daily_ingestion


def main():
    """Start the Prefect worker serving the daily ingestion flow."""
    deployment = daily_ingestion.to_deployment(
        name="daily-ingestion",
        cron="0 6 * * *",
        tags=["rag", "ingestion"],
    )
    serve(deployment)


if __name__ == "__main__":
    main()
