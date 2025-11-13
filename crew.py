from dotenv import load_dotenv
import yaml
import os
from pathlib import Path

# Only load .env file if OPENAI_API_KEY is not already set
# On Streamlit Cloud, streamlit_app.py sets it from secrets before importing this module
# This ensures .env is NEVER used on Streamlit Cloud, only for local development
if "OPENAI_API_KEY" not in os.environ:
    load_dotenv()

from crewai import Crew, Process, Agent, Task
from langchain_openai import ChatOpenAI
from agent.tools.pdf_extractor_fitz import PdfExtractorFitz
from agent.tools.vendor_master_lookup import VendorMasterLookup
from agent.tools.cost_category_lookup import CostCategoryLookup
from agent.tools.vat_calculator import VATCalculator

# Configure LLM explicitly for CrewAI
# This ensures CrewAI can find and use the OpenAI provider
# The API key should already be set in os.environ by streamlit_app.py
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "OPENAI_API_KEY not found in environment. "
        "Please set it in Streamlit secrets or .env file."
    )

llm = ChatOpenAI(
    model="gpt-4o",  # or "gpt-3.5-turbo" for faster/cheaper
    temperature=0.7,
)

# Get base directory for deployment-safe paths
BASE_DIR = Path(__file__).resolve().parent

# Instantiate tools with their BaseTool "config" fields using absolute paths
pdf_tool = PdfExtractorFitz(max_pages=0)  
master_tool = VendorMasterLookup(master_path=str(BASE_DIR / "knowledge" / "Vendor-Master_20251030_total.xlsx"), min_score=0.82)
cost_category_tool = CostCategoryLookup(allocation_path=str(BASE_DIR / "knowledge" / "Vendor_Cost-Category-Allocation_V20251031.xlsx"))
vat_calculator_tool = VATCalculator()

# Load agents from YAML using absolute path
agent_config_path = BASE_DIR / "agent" / "config" / "agent.yaml"
with open(agent_config_path, "r", encoding="utf-8") as f:
    agents_data = yaml.safe_load(f)

agents = []
agents_by_name = {}  # Dictionary to map agent names to agent objects
for agent_data in agents_data["agents"]:
    # Assign tools based on agent name
    agent_name = agent_data["name"]
    if agent_name == "Vendor_Basic_Info_Agent":
        # Vendor_Basic_Info_Agent needs all 4 tools
        agent_tools = [pdf_tool, master_tool, cost_category_tool, vat_calculator_tool]
    elif agent_name == "Invoice_Details_Agent":
        # Invoice_Details_Agent needs only pdf_tool and vat_calculator_tool
        agent_tools = [pdf_tool, vat_calculator_tool]
    elif agent_name == "Result_Merger_Agent":
        # Result_Merger_Agent needs no tools (just merges JSON)
        agent_tools = []
    else:
        # Default: all tools
        agent_tools = [pdf_tool, master_tool, cost_category_tool, vat_calculator_tool]
    
    agent = Agent(
        name=agent_data["name"],
        role=agent_data["role"],
        goal=agent_data["goal"],
        backstory=agent_data["backstory"],
        allow_delegation=agent_data.get("allow_delegation", False),
        verbose=agent_data.get("verbose", True),
        memory=agent_data.get("memory", False),
        tools=agent_tools,
        llm=llm  # Explicitly set the LLM
    )
    agents.append(agent)
    agents_by_name[agent_data["name"]] = agent  # Store mapping

# Load tasks from YAML using absolute path
tasks_config_path = BASE_DIR / "agent" / "config" / "tasks.yaml"
with open(tasks_config_path, "r", encoding="utf-8") as f:
    tasks_data = yaml.safe_load(f)

tasks = []
task_objects = {}  # Dictionary to store task objects for dependencies

# First pass: Create all tasks
for task_name, task_data in tasks_data.items():
    # Find the agent by name using the dictionary
    agent = agents_by_name.get(task_data["agent"])
    if agent:
        task = Task(
            description=task_data["description"],
            expected_output=task_data["expected_output"],
            agent=agent
        )
        tasks.append(task)
        task_objects[task_name] = task

# Second pass: Set up context dependencies for merge_results_task
if "merge_results_task" in task_objects:
    vendor_task = task_objects.get("extract_vendor_basic_info_task")
    invoice_task = task_objects.get("extract_invoice_details_task")
    
    if vendor_task and invoice_task:
        # Update the merge task with context from previous tasks
        merge_task = task_objects["merge_results_task"]
        # In CrewAI, we need to recreate the task with context
        # Find the merge task in the tasks list and update it
        for i, task in enumerate(tasks):
            if task == merge_task:
                # Recreate the task with context
                agent = agents_by_name.get(tasks_data["merge_results_task"]["agent"])
                tasks[i] = Task(
                    description=tasks_data["merge_results_task"]["description"],
                    expected_output=tasks_data["merge_results_task"]["expected_output"],
                    agent=agent,
                    context=[vendor_task, invoice_task]
                )
                task_objects["merge_results_task"] = tasks[i]
                break

crew = Crew(
    agents=agents,
    tasks=tasks,
    process=Process.sequential
)

def run(pdf_path: str):
    # kickoff executes tasks; agent will call tools by name
    result = crew.kickoff(inputs={"pdf_path": pdf_path})
    print(result)
    return result

if __name__ == "__main__":
    run("ExampleFiles/Attachments_OCR-Project (1)/CopyBlitz_re-2025-1535.pdf")
