from mcp.server.fastmcp import FastMCP
from LMOTask import LMOTask

mcp = FastMCP("Hypothesis MCP Server")

# Instance of the class
obj = LMOTask()


@mcp.tool()
def check_final_proposal(smiles_list_as_string: str) -> bool:
    """
    Check if the proposed SMILES strings are valid and meet the criteria.
    The criteria are:
    1. The SMILES must be valid.
    2. The synthesizability score must be less than or equal to the lead molecule.
    3. The density must be greater than or equal to the lead molecule.

    Args:
        smiles (str): The proposed  list of SMILES strings.
    Returns:
        bool: True if the proposal is valid and meets the criteria, False otherwise.

    Raises:
        ValueError: If the output is not a valid list of SMILES strings or if any
                    SMILES string is invalid or does not meet the criteria.
    """
    return obj.check_final_proposal(smiles_list_as_string=smiles_list_as_string)


if __name__ == "__main__":
    mcp.run(transport="stdio")
