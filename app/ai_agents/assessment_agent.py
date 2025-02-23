from typing import Dict, List, Any, Tuple
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
import json
import re

class DataAssessment(BaseModel):
    """Pydantic model for structured data assessment output."""
    quality_score: float = Field(
        description="Overall data quality score (0-1)",
        ge=0.0,
        le=1.0
    )
    issues_detected: List[str] = Field(
        description="List of identified data quality issues"
    )
    cleaning_recommendations: List[Dict[str, Any]] = Field(
        description="List of recommended cleaning actions with parameters"
    )
    column_specific_issues: Dict[str, List[str]] = Field(
        description="Issues specific to individual columns",
        default_factory=dict
    )
    column_classifications: Dict[str, str] = Field(
        description="Classification of each column (e.g., 'identifier', 'categorical', 'numeric', 'datetime', etc.)",
        default_factory=dict
    )
    critical_columns: List[str] = Field(
        description="List of columns that should not be modified/filled",
        default_factory=list
    )

class DataAssessmentAgent:
    """Agent for analyzing datasets and providing cleaning recommendations."""

    def __init__(self):
        """Initialize the assessment agent with Google's Gemini model."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            google_api_key=api_key
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data quality expert. Your task is to analyze datasets and determine their structure, patterns, and relationships WITHOUT making assumptions about column names or types.

For each column, analyze:
1. The distribution and patterns in the data
2. The uniqueness of values
3. The relationships between values
4. The semantic meaning based on the actual content (not just names)
5. Whether the column contains critical/sensitive information based on value patterns

Identify columns as critical if they:
- Contain unique identifiers
- Show patterns typical of personal information
- Have high cardinality with non-numeric patterns
- Appear to be key reference data
- Show date/timestamp patterns
- Contain what appears to be measurement or transaction data

Your response must follow this EXACT format:

QUALITY SCORE: [number between 0 and 1]

ISSUES DETECTED:
- [issue 1]
- [issue 2]
...

CLEANING RECOMMENDATIONS:
1. Action: [action name]
   Parameters: [parameters as key-value pairs]
2. Action: [action name]
   Parameters: [parameters as key-value pairs]
...

COLUMN CLASSIFICATIONS:
[column name]: [classification based on content analysis]
...

CRITICAL COLUMNS:
- [column 1 with explanation of why it's critical]
- [column 2 with explanation of why it's critical]
...

COLUMN SPECIFIC ISSUES:
[column name]:
- [issue 1]
- [issue 2]
...
"""),
            ("human", """
            Please analyze this dataset and provide a quality assessment in the required format.
            Focus on the actual data patterns and content, not column names.
            
            Dataset Information:
            {dataset_info}
            
            Sample Data:
            {sample_data}
            
            Column Statistics:
            {column_stats}
            """)
        ])

    def _analyze_column_patterns(self, df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
        """Analyze patterns in each column to determine their type and criticality."""
        classifications = {}
        critical_columns = []
        
        for column in df.columns:
            sample_values = df[column].dropna().head(10).astype(str).tolist()
            dtype = str(df[column].dtype)
            unique_ratio = df[column].nunique() / len(df)
            
            # Analyze patterns in the data
            if dtype in ['int64', 'float64']:
                if unique_ratio > 0.9:  # Highly unique numbers might be IDs
                    classifications[column] = 'identifier'
                    critical_columns.append(column)
                else:
                    classifications[column] = 'numeric'
            elif 'datetime' in dtype:
                classifications[column] = 'datetime'
                critical_columns.append(column)
            else:
                # Check for common patterns in string values
                if any(self._looks_like_email(str(v)) for v in sample_values):
                    classifications[column] = 'email'
                    critical_columns.append(column)
                elif any(self._looks_like_phone(str(v)) for v in sample_values):
                    classifications[column] = 'phone'
                    critical_columns.append(column)
                elif unique_ratio > 0.8:  # Highly unique strings might be names/identifiers
                    classifications[column] = 'identifier'
                    critical_columns.append(column)
                else:
                    classifications[column] = 'categorical'
        
        return classifications, critical_columns

    def _looks_like_email(self, value: str) -> bool:
        """Check if a value looks like an email address."""
        return '@' in value and '.' in value.split('@')[1]

    def _looks_like_phone(self, value: str) -> bool:
        """Check if a value looks like a phone number."""
        digits = ''.join(filter(str.isdigit, value))
        return len(digits) >= 7

    def _generate_dataset_info(self, df: pd.DataFrame) -> str:
        """Generate a text summary of the dataset."""
        info = f"""
        Shape: {df.shape}
        Columns: {', '.join(df.columns)}
        Data Types: {df.dtypes.to_dict()}
        Missing Values: {df.isna().sum().to_dict()}
        Duplicate Rows: {df.duplicated().sum()}
        """
        return info

    def _generate_column_stats(self, df: pd.DataFrame) -> str:
        """Generate statistical information about each column."""
        stats = []
        
        for column in df.columns:
            col_stats = {
                'name': column,
                'dtype': str(df[column].dtype),
                'unique_values': df[column].nunique(),
                'missing_values': df[column].isna().sum()
            }
            
            if df[column].dtype in ['int64', 'float64']:
                col_stats.update({
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max())
                })
            elif df[column].dtype == 'object':
                col_stats['sample_values'] = df[column].dropna().sample(
                    min(5, df[column].nunique())
                ).tolist()
            
            stats.append(col_stats)
        
        return str(stats)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format."""
        try:
            # Extract quality score
            quality_match = re.search(r'QUALITY SCORE:\s*(0?\.\d+|1\.0?|1|0)', content)
            quality_score = float(quality_match.group(1)) if quality_match else 0.5

            # Extract issues detected
            issues_section = re.search(r'ISSUES DETECTED:(.*?)(?=CLEANING RECOMMENDATIONS:|$)', content, re.DOTALL)
            issues = []
            if issues_section:
                issues = [issue.strip('- \n') for issue in issues_section.group(1).strip().split('\n') if issue.strip('- \n')]

            # Extract cleaning recommendations
            recommendations = []
            recommendations_section = re.search(r'CLEANING RECOMMENDATIONS:(.*?)(?=COLUMN CLASSIFICATIONS:|$)', content, re.DOTALL)
            if recommendations_section:
                current_action = None
                current_params = {}
                for line in recommendations_section.group(1).strip().split('\n'):
                    line = line.strip()
                    if 'Action:' in line:
                        if current_action:
                            recommendations.append({"action": current_action, "parameters": current_params})
                        current_action = line.split('Action:')[1].strip()
                        current_params = {}
                    elif 'Parameters:' in line:
                        params_str = line.split('Parameters:')[1].strip()
                        try:
                            current_params = eval(f"dict({params_str})")
                        except:
                            current_params = {}
                if current_action:
                    recommendations.append({"action": current_action, "parameters": current_params})

            # Extract column classifications
            classifications = {}
            classifications_section = re.search(r'COLUMN CLASSIFICATIONS:(.*?)(?=CRITICAL COLUMNS:|$)', content, re.DOTALL)
            if classifications_section:
                for line in classifications_section.group(1).strip().split('\n'):
                    if ':' in line:
                        col, cls = line.split(':', 1)
                        classifications[col.strip()] = cls.strip()

            # Extract critical columns
            critical_columns = []
            critical_section = re.search(r'CRITICAL COLUMNS:(.*?)(?=COLUMN SPECIFIC ISSUES:|$)', content, re.DOTALL)
            if critical_section:
                critical_columns = [col.strip('- \n') for col in critical_section.group(1).strip().split('\n') if col.strip('- \n')]

            # Extract column specific issues
            column_issues = {}
            column_section = re.search(r'COLUMN SPECIFIC ISSUES:(.*?)$', content, re.DOTALL)
            if column_section:
                current_column = None
                current_issues = []
                for line in column_section.group(1).strip().split('\n'):
                    line = line.strip()
                    if line.endswith(':'):
                        if current_column and current_issues:
                            column_issues[current_column] = current_issues
                        current_column = line.rstrip(':')
                        current_issues = []
                    elif line.startswith('- '):
                        current_issues.append(line.strip('- '))
                if current_column and current_issues:
                    column_issues[current_column] = current_issues

            return {
                "quality_score": quality_score,
                "issues_detected": issues,
                "cleaning_recommendations": recommendations,
                "column_classifications": classifications,
                "critical_columns": critical_columns,
                "column_specific_issues": column_issues
            }
        except Exception as e:
            return {
                "quality_score": 0.5,
                "issues_detected": ["Failed to parse AI response"],
                "cleaning_recommendations": [
                    {"action": "standardize_names", "parameters": {}},
                    {"action": "handle_missing", "parameters": {"strategy": "auto"}},
                    {"action": "remove_duplicates", "parameters": {}}
                ],
                "column_classifications": {},
                "critical_columns": [],
                "column_specific_issues": {}
            }

    async def assess_dataset(self, df: pd.DataFrame) -> DataAssessment:
        """
        Analyze the dataset and provide cleaning recommendations.
        
        Args:
            df: The pandas DataFrame to analyze
            
        Returns:
            DataAssessment object containing the analysis and recommendations
        """
        # Analyze column patterns
        classifications, critical_columns = self._analyze_column_patterns(df)
        
        # Prepare the dataset information
        dataset_info = self._generate_dataset_info(df)
        sample_data = df.head().to_string()
        column_stats = self._generate_column_stats(df)
        
        # Create the prompt
        prompt = self.prompt.format_messages(
            dataset_info=dataset_info,
            sample_data=sample_data,
            column_stats=column_stats
        )
        
        # Get the assessment from the LLM
        response = await self.llm.ainvoke(prompt)
        
        # Parse the response using our custom parser
        assessment_dict = self._parse_response(response.content)
        
        # Combine AI's assessment with our pattern analysis
        assessment_dict["column_classifications"].update(classifications)
        assessment_dict["critical_columns"] = list(set(assessment_dict["critical_columns"] + critical_columns))
        
        # Create a DataAssessment object
        assessment = DataAssessment(
            quality_score=assessment_dict["quality_score"],
            issues_detected=assessment_dict["issues_detected"],
            cleaning_recommendations=assessment_dict["cleaning_recommendations"],
            column_classifications=assessment_dict["column_classifications"],
            critical_columns=assessment_dict["critical_columns"],
            column_specific_issues=assessment_dict["column_specific_issues"]
        )
        
        return assessment

    def get_cleaning_strategy(self, assessment: DataAssessment) -> Dict[str, Any]:
        """
        Convert assessment recommendations into cleaning parameters.
        
        Args:
            assessment: DataAssessment object from assess_dataset
            
        Returns:
            Dictionary of cleaning strategy parameters
        """
        strategy = {
            'standardize_names': True,
            'remove_duplicates': True,
            'handle_missing': 'auto',
            'detect_outliers': 3.0,
            'convert_types': True,
            'remove_low_variance': 0.01,
            'critical_columns': assessment.critical_columns
        }
        
        # Update strategy based on recommendations
        for recommendation in assessment.cleaning_recommendations:
            action = recommendation.get('action', '').lower()
            params = recommendation.get('parameters', {})
            
            if action == 'handle_missing':
                strategy['handle_missing'] = params.get('strategy', 'auto')
            elif action == 'detect_outliers':
                strategy['detect_outliers'] = params.get('threshold', 3.0)
            elif action == 'remove_low_variance':
                strategy['remove_low_variance'] = params.get('threshold', 0.01)
        
        return strategy 