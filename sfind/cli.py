import asyncio

import click
from rich.console import Console
from rich.table import Table

from sfind.config.config import create_context
from sfind.models.models import RetrieveRequest, RetrieveResponse, TYPE_REGISTER
from sfind.core.orchestrator import Orchestrator


@click.command()
@click.option('--query', help='Text query to search against')
@click.option('--path', help='Path to search against')
@click.option('--file_type', help='Types of files to search')
@click.option("--limit",default = 5, help="Show only top N files")
@click.option("--explain",default = False, help="Explain why this file appeared in the search result")
def main(query: str, path: str, file_type: str, limit: int, explain: bool):
    asyncio.run(run_async(query, path, file_type, limit, explain))

async def run_async(query: str, path: str, file_type: str, limit: int, explain: bool):
    context = create_context()
    response = await Orchestrator(context=context).execute(request=_get_retrieve_request(query=query, path=path, file_type=file_type, explain=explain, limit=limit))
    show_output(query, response, explain)

def _get_retrieve_request(query: str, path: str, file_type: str, explain: bool, limit: int) -> RetrieveRequest:
    return RetrieveRequest(
        prompt=query,
        path=path,
        file_types=TYPE_REGISTER[file_type],
        explain=explain,
        limit=limit
    )

def show_output(query: str, response: list[RetrieveResponse], explain: bool=False) -> None:
    table = Table(title=f"Results for query: {query}")

    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("File", style="magenta", no_wrap=False)
    table.add_column("Score", justify="right", style="green")
    if explain is True:
        table.add_column("Caption", justify="right", style="green")

    for index, response_item in enumerate(response, start=1):
        if explain is False:
            table.add_row(str(index), response_item.file_uri, f"{response_item.similarity_score:.3f}")
        else:
            table.add_row(str(index), response_item.file_uri, f"{response_item.similarity_score:.3f}", response_item.description)

    console = Console()
    console.print(table)
if __name__ == "__main__":
    main()

