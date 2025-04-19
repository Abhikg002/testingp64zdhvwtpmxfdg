#Working for employee vs account

import pandas as pd
from typing import Dict, Optional
from src.bedrock_llm import invoke_claude_model
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

def filter_open_demands(demand_df: pd.DataFrame, demand_col_map: Dict[str, str], selected_account: Optional[str] = None) -> pd.DataFrame:
    """
    Filters open demands using mapped column for 'Position Status' and (optionally) by selected account.
    """
    status_col = demand_col_map["Position Status"]
    open_demands = demand_df[demand_df[status_col].astype(str).str.strip().str.lower() == "open"]

    if selected_account:
        account_col = demand_col_map["Account"]
        open_demands = open_demands[open_demands[account_col].astype(str).str.strip() == selected_account.strip()]

    return open_demands


def get_matching_demands_for_employee(employee: Dict[str, str], open_demands: pd.DataFrame, emp_col_map: Dict[str, str], demand_col_map: Dict[str, str], top_n=10) -> pd.DataFrame:
    """
    Filters top N open demands for an employee based on keyword overlap.
    """
    emp_skills = str(employee.get(emp_col_map["Detailed Skill view"], "")).lower().split(",")
    emp_skills = [skill.strip() for skill in emp_skills]

    def skill_overlap(row):
        demand_skills = str(row.get(demand_col_map["Skillset"], "")).lower()
        return sum(1 for skill in emp_skills if skill in demand_skills)

    open_demands = open_demands.copy()
    open_demands["match_score_temp"] = open_demands.apply(skill_overlap, axis=1)
    return open_demands.sort_values("match_score_temp", ascending=False).head(top_n).drop(columns=["match_score_temp"])


def build_prompt(employee: Dict[str, str], open_demands: pd.DataFrame, emp_col_map: Dict[str, str], demand_col_map: Dict[str, str]) -> str:
    """
    Builds Claude prompt using dynamic mappings.
    """
    emp_details = f"""
    Employee Profile:
    - ID: {employee.get(emp_col_map['Emp ID'])}
    - Name: {employee.get(emp_col_map['Name'])}
    - Grade: {employee.get(emp_col_map['Grade'])}
    - City: {employee.get(emp_col_map['City'])}
    - Skillset: {employee.get(emp_col_map['Detailed Skill view'])}
    - GTD Skill: {employee.get(emp_col_map['GTD skill'])}
    """

    demand_descriptions = ""
    for _, row in open_demands.iterrows():
        demand_descriptions += f"""
        Demand ID: {row.get(demand_col_map['Demand ID'], '')}
        Skillset: {row.get(demand_col_map['Skillset'], '')}
        Grade: {row.get(demand_col_map['Grade'], '')}
        Location: {row.get(demand_col_map['Location'], '')}
        Account: {row.get(demand_col_map['Account'], '')}
        """

    prompt = f"""
    You are an AI assistant helping with internal resource allocation. Match the below employee profile
    with the most suitable open demands based on skillset, grade, and location.

    Provide a list of best-matching demands sorted by relevance with:
    - Match Score (0-100)
    - Matching Skills
    - Account Name

    {emp_details}

    Open Demands:
    {demand_descriptions}
    """

    return prompt


def parse_matches(response_text: str):
    """
    Parses Claude response to extract structured match data.
    """
    matches = []
    pattern = re.compile(
        r"(?i)demand id: ([^\n]+).*?match score: (\d+).*?matching skills: ([^\n]+).*?account: ([^\n]+)",
        re.DOTALL
    )

    for match in pattern.findall(response_text):
        demand_id, score, skills, account = match
        matches.append({
            "Demand ID": demand_id.strip(),
            "Match Score": int(score),
            "Matching Skills": skills.strip(),
            "Account": account.strip()
        })

    return matches




@st.cache_data(show_spinner=False)
def match_employees_to_demands(employee_df: pd.DataFrame,
                                demand_df: pd.DataFrame,
                                emp_col_map: Dict[str, str],
                                demand_col_map: Dict[str, str],
                                selected_account: Optional[str],
                                aws_access_key: str,
                                aws_secret_key: str,
                                aws_region: str) -> pd.DataFrame:
    """
    Master function to match employees to filtered open demands.
    Uses dynamic column mapping and optional account filter.
    """
    open_demands = filter_open_demands(demand_df, demand_col_map, selected_account)
    all_matches = []

    def process_employee(emp_row):
        employee = emp_row.to_dict()
        top_demands = get_matching_demands_for_employee(employee, open_demands, emp_col_map, demand_col_map)
        prompt = build_prompt(employee, top_demands, emp_col_map, demand_col_map)
        response_text = invoke_claude_model(prompt, aws_access_key, aws_secret_key, aws_region)
        parsed_matches = parse_matches(response_text)

        results = []
        for match in parsed_matches:
            results.append({
                "Employee Name": employee.get(emp_col_map["Name"]),
                "Emp ID": employee.get(emp_col_map["Emp ID"]),
                "Grade": employee.get(emp_col_map["Grade"]),
                "City": employee.get(emp_col_map["City"]),
                "Detailed Skill view": employee.get(emp_col_map["Detailed Skill view"]),
                "GTD skill": employee.get(emp_col_map["GTD skill"]),
                **match
            })
        return results

    progress_placeholder = st.empty()
    all_matches = []
    futures = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        for idx, (_, emp_row) in enumerate(employee_df.iterrows(), 1):
            future = executor.submit(process_employee, emp_row)
            futures.append((idx, emp_row, future))

        for idx, emp_row, future in futures:
            employee_name = emp_row.get(emp_col_map["Name"])
            progress_placeholder.info(f"🔄 Matching employee {idx}/{len(futures)}: **{employee_name}**")
            all_matches.extend(future.result())

    progress_placeholder.success("✅ All employees processed.")


    return pd.DataFrame(all_matches)
