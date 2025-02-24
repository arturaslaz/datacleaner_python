import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import datetime
import json
from enum import Enum
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import os
import re
from app.ai_agents.assessment_agent import DataAssessment

class AICleaningOperations:
    """Class for AI-powered data cleaning operations."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with a language model."""
        self.llm = llm
        self.confidence_threshold = 0.8
        
        # Initialize prompts
        self.imputation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data cleaning expert. Analyze the context and suggest an appropriate value for the missing data.
            Only suggest a value if you are highly confident it is correct. Otherwise, return NULL."""),
            ("human", """Given the following context about a missing value:
            Column: {column_name}
            Column Type: {column_type}
            Row Data: {row_data}
            Column Statistics: {stats}
            Similar Complete Rows: {examples}

            Suggest the most appropriate value based on:
            1. Patterns in other columns
            2. Statistical properties
            3. Business logic and data consistency

            Respond ONLY with:
            VALUE: suggested_value or NULL
            CONFIDENCE: 0-1
            EXPLANATION: Brief justification""")
        ])
        
        self.anomaly_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data quality expert. Analyze potentially anomalous values and suggest corrections only when highly confident."""),
            ("human", """Analyze this potentially anomalous value:
            Value: {value}
            Column: {column_name}
            Context: {context}
            Pattern Violations: {violations}
            Valid Examples: {examples}

            Respond with:
            ANOMALY: yes/no
            CORRECTION: suggested_value or REMOVE
            CONFIDENCE: 0-1
            EXPLANATION: Brief justification""")
        ])
        
        self.standardization_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data standardization expert. Suggest standardized forms for similar values while preserving semantic meaning."""),
            ("human", """Analyze these similar values in column '{column_name}':
            Values: {value_list}
            Frequencies: {frequency_dict}
            Detected Pattern: {pattern}

            Respond with:
            STANDARD_FORM: suggested_standardization
            CONFIDENCE: 0-1
            EXPLANATION: Brief justification""")
        ])
        
        self.type_inference_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data type expert. Analyze values to determine the most appropriate data type and conversion method."""),
            ("human", """Analyze these values for type inference:
            Column: {column_name}
            Current Type: {current_type}
            Sample Values: {sample_values}
            Semantic Hints: {semantic_hints}

            Respond with:
            TYPE: inferred_type
            CONVERSION: conversion_method
            CONFIDENCE: 0-1
            EXPLANATION: Brief justification""")
        ])

    async def _parse_ai_response(self, response: str, expected_keys: List[str]) -> Dict[str, Any]:
        """Parse AI response into a structured format."""
        result = {}
        for line in response.strip().split('\n'):
            for key in expected_keys:
                if line.upper().startswith(f"{key}:"):
                    result[key.lower()] = line.split(':', 1)[1].strip()
        return result

    async def ai_impute_value(self, column: str, row_context: Dict) -> Any:
        """Use AI to impute missing values based on context."""
        try:
            # Format the context for the prompt
            prompt_context = {
                "column_name": column,
                "column_type": row_context.get("column_stats", {}).get("type", "unknown"),
                "row_data": json.dumps(row_context["row_data"]),
                "stats": json.dumps(row_context["column_stats"]),
                "examples": json.dumps(row_context["pattern_examples"])
            }
            
            # Get AI suggestion
            response = await self.llm.ainvoke(
                self.imputation_prompt.format_messages(**prompt_context)
            )
            
            # Parse response
            result = await self._parse_ai_response(
                response.content,
                ["VALUE", "CONFIDENCE", "EXPLANATION"]
            )
            
            # Return value only if confidence is high enough
            confidence = float(result.get("confidence", 0))
            if confidence >= self.confidence_threshold:
                value = result.get("value")
                return None if value == "NULL" else value
            return None
            
        except Exception as e:
            print(f"Error in AI imputation: {str(e)}")
            return None

    async def ai_correct_anomalies(self, column: str, suspicious_values: List[Dict]) -> Dict[Any, Any]:
        """Use AI to detect and correct anomalous values."""
        corrections = {}
        try:
            for value_info in suspicious_values:
                prompt_context = {
                    "value": value_info["value"],
                    "column_name": column,
                    "context": json.dumps(value_info["context"]),
                    "violations": json.dumps(value_info["pattern_violations"]),
                    "examples": json.dumps(value_info.get("valid_examples", []))
                }
                
                response = await self.llm.ainvoke(
                    self.anomaly_prompt.format_messages(**prompt_context)
                )
                
                result = await self._parse_ai_response(
                    response.content,
                    ["ANOMALY", "CORRECTION", "CONFIDENCE", "EXPLANATION"]
                )
                
                if (result.get("anomaly", "").lower() == "yes" and 
                    float(result.get("confidence", 0)) >= self.confidence_threshold):
                    correction = result.get("correction")
                    if correction and correction != "REMOVE":
                        corrections[value_info["value"]] = correction
                    elif correction == "REMOVE":
                        corrections[value_info["value"]] = None
                        
            return corrections
            
        except Exception as e:
            print(f"Error in anomaly correction: {str(e)}")
            return {}

    async def ai_standardize_values(self, column: str, value_groups: List[Dict]) -> Dict[str, str]:
        """Use AI to standardize similar values."""
        standardizations = {}
        try:
            for group in value_groups:
                prompt_context = {
                    "column_name": column,
                    "value_list": json.dumps(group["values"]),
                    "frequency_dict": json.dumps(group["frequency"]),
                    "pattern": group.get("pattern", "")
                }
                
                response = await self.llm.ainvoke(
                    self.standardization_prompt.format_messages(**prompt_context)
                )
                
                result = await self._parse_ai_response(
                    response.content,
                    ["STANDARD_FORM", "CONFIDENCE", "EXPLANATION"]
                )
                
                if float(result.get("confidence", 0)) >= self.confidence_threshold:
                    standard_form = result.get("standard_form")
                    if standard_form:
                        for value in group["values"]:
                            standardizations[value] = standard_form
                            
            return standardizations
            
        except Exception as e:
            print(f"Error in value standardization: {str(e)}")
            return {}

    async def ai_infer_and_convert_type(self, column: str, sample_data: Dict) -> Tuple[str, Any]:
        """Use AI to infer and convert data types based on content and context."""
        try:
            prompt_context = {
                "column_name": column,
                "current_type": sample_data["current_type"],
                "sample_values": json.dumps(sample_data["values"]),
                "semantic_hints": json.dumps(sample_data.get("semantic_hints", {}))
            }
            
            response = await self.llm.ainvoke(
                self.type_inference_prompt.format_messages(**prompt_context)
            )
            
            result = await self._parse_ai_response(
                response.content,
                ["TYPE", "CONVERSION", "CONFIDENCE", "EXPLANATION"]
            )
            
            if float(result.get("confidence", 0)) >= self.confidence_threshold:
                inferred_type = result.get("type")
                conversion_method = result.get("conversion")
                
                # Create conversion function based on the suggested method
                if conversion_method and inferred_type:
                    def conversion_func(x):
                        try:
                            if pd.isna(x):
                                return x
                            if inferred_type == "datetime":
                                return pd.to_datetime(x)
                            elif inferred_type in ["int", "float"]:
                                return pd.to_numeric(str(x).replace(",", ""))
                            elif inferred_type == "boolean":
                                return str(x).lower() in ["true", "1", "yes", "y"]
                            else:
                                return str(x)
                        except:
                            return x
                            
                    return inferred_type, conversion_func
                    
            return None, None
            
        except Exception as e:
            print(f"Error in type inference: {str(e)}")
            return None, None

class AIDecisionType(Enum):
    """Types of decisions made by AI during cleaning."""
    IMPUTATION = "imputation"
    ANOMALY_CORRECTION = "anomaly_correction"
    TYPE_CONVERSION = "type_conversion"
    STANDARDIZATION = "standardization"
    PATTERN_CORRECTION = "pattern_correction"

@dataclass
class AIDecision:
    """Record of an AI decision during cleaning."""
    decision_type: AIDecisionType
    column: str
    original_value: Any
    suggested_value: Any
    confidence: float
    explanation: str
    applied: bool
    validation_result: Optional[bool] = None
    validation_message: Optional[str] = None

@dataclass
class AICleaningContext:
    """Context information for AI cleaning operations."""
    column_patterns: Dict[str, List[str]]
    value_relationships: Dict[str, Dict[str, float]]
    detected_formats: Dict[str, str]
    confidence_thresholds: Dict[str, float]
    cleaning_history: List[AIDecision]

@dataclass
class CleaningReport:
    """Class to store cleaning operation results and statistics."""
    original_shape: Tuple[int, int]
    cleaned_shape: Tuple[int, int]
    missing_values_filled: Dict[str, int]
    duplicates_removed: int
    outliers_detected: Dict[str, List[int]]
    column_types_converted: Dict[str, Tuple[str, str]]
    columns_dropped: List[str]
    standardized_columns: List[str]
    cleaning_steps: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    ai_decisions: List[AIDecision]  # New field for AI decisions
    cleaning_context: AICleaningContext  # New field for cleaning context

class DataCleaner:
    """Main class for data cleaning operations."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a copy of the original dataframe."""
        self.original_df = df.copy()
        self.df = df.copy()
        self.report = CleaningReport(
            original_shape=df.shape,
            cleaned_shape=df.shape,
            missing_values_filled={},
            duplicates_removed=0,
            outliers_detected={},
            column_types_converted={},
            columns_dropped=[],
            standardized_columns=[],
            cleaning_steps=[],
            validation_results=[],
            ai_decisions=[],
            cleaning_context=AICleaningContext(
                column_patterns={},
                value_relationships={},
                detected_formats={},
                confidence_thresholds={},
                cleaning_history=[]
            )
        )

    def standardize_column_names(self) -> None:
        """Standardize column names to snake_case."""
        original_columns = self.df.columns.tolist()
        self.df.columns = (self.df.columns
                          .str.lower()
                          .str.replace(r'[^\w\s]', '', regex=True)
                          .str.replace(r'\s+', '_', regex=True))
        self.report.standardized_columns = [
            f"{old} → {new}" for old, new in zip(original_columns, self.df.columns)
            if old != new
        ]

    def remove_duplicates(self) -> None:
        """Remove duplicate rows from the dataset."""
        original_count = len(self.df)
        self.df = self.df.drop_duplicates(keep='first')
        self.report.duplicates_removed = original_count - len(self.df)
        self.report.cleaned_shape = self.df.shape

    def _is_valid_date(self, value: str) -> bool:
        """Check if a string can be parsed as a valid date."""
        try:
            if pd.isna(value) or value == '':
                return False
            pd.to_datetime(str(value))
            return True
        except:
            return False

    def _convert_to_numeric(self, series: pd.Series) -> Tuple[pd.Series, bool]:
        """Try to convert a series to numeric, handling common formats."""
        # Remove any currency symbols and commas
        if series.dtype == 'object':
            series = series.str.replace(r'[,$€£]', '', regex=True)
            series = series.str.replace(',', '')
        
        # Try converting to numeric
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            # Only consider it a successful conversion if most values were converted
            success = numeric_series.notna().sum() / len(numeric_series) > 0.5
            return numeric_series, success
        except:
            return series, False

    def handle_missing_values(self, strategy: str = 'auto', critical_columns: List[str] = None) -> None:
        """
        Handle missing values in the dataset based on AI assessment.
        Never create new data - only use existing values if there's an exact match pattern.
        
        Args:
            strategy: 'auto' (choose based on column type), 'drop', or 'fill'
            critical_columns: List of columns identified as critical by AI assessment
        """
        critical_columns = critical_columns or []
        
        for column in self.df.columns:
            missing_count = self.df[column].isna().sum()
            if missing_count == 0 or column in critical_columns:
                continue

            if strategy == 'auto':
                # Get non-null values for this column
                non_null_values = self.df[column].dropna()
                
                # Only proceed if we have non-null values
                if len(non_null_values) == 0:
                    continue
                
                # For any type of column, only fill if we can find an exact match
                # based on other columns' values
                for idx in self.df[self.df[column].isna()].index:
                    row = self.df.loc[idx]
                    
                    # Find rows with the same values in other columns
                    mask = pd.Series(True, index=self.df.index)
                    for other_col in self.df.columns:
                        if other_col != column and not pd.isna(row[other_col]):
                            mask &= (self.df[other_col] == row[other_col])
                    matching_rows = self.df[mask & ~self.df[column].isna()]
                    
                    # Only fill if we find exactly one matching pattern
                    if len(matching_rows) == 1:
                        self.df.at[idx, column] = matching_rows[column].iloc[0]
                        self.report.missing_values_filled[column] = self.report.missing_values_filled.get(column, 0) + 1
            
            elif strategy == 'drop' and column not in critical_columns:
                original_len = len(self.df)
                self.df = self.df.dropna(subset=[column])
                dropped_count = original_len - len(self.df)
                if dropped_count > 0:
                    self.report.missing_values_filled[column] = dropped_count

        self.report.cleaned_shape = self.df.shape

    def detect_outliers(self, threshold: float = 3.0) -> None:
        """
        Detect outliers using z-score method.
        
        Args:
            threshold: z-score threshold for outlier detection
        """
        numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        for column in numeric_columns:
            z_scores = np.abs(StandardScaler().fit_transform(self.df[[column]]))
            outlier_indices = np.where(z_scores > threshold)[0]
            
            if len(outlier_indices) > 0:
                self.report.outliers_detected[column] = outlier_indices.tolist()

    def convert_data_types(self) -> None:
        """Convert columns to appropriate data types."""
        for column in self.df.columns:
            original_type = str(self.df[column].dtype)
            
            # Skip conversion if already numeric or datetime
            if self.df[column].dtype in ['int64', 'float64', 'datetime64[ns]']:
                continue

            # Try converting to numeric first
            numeric_series, is_numeric = self._convert_to_numeric(self.df[column])
            if is_numeric:
                self.df[column] = numeric_series
                self.report.column_types_converted[column] = (original_type, str(self.df[column].dtype))
                continue

            # Try converting to datetime
            if self.df[column].dtype == 'object':
                # Check if most values in the column look like dates
                valid_dates = self.df[column].apply(self._is_valid_date)
                if valid_dates.sum() / len(valid_dates) > 0.5:
                    try:
                        self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                        self.report.column_types_converted[column] = (original_type, 'datetime64[ns]')
                    except:
                        pass

    def remove_low_variance_columns(self, threshold: float = 0.01) -> None:
        """
        Remove columns with very low variance.
        
        Args:
            threshold: minimum variance threshold
        """
        numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        for column in numeric_columns:
            if self.df[column].var() < threshold:
                self.df = self.df.drop(columns=[column])
                self.report.columns_dropped.append(column)
        
        self.report.cleaned_shape = self.df.shape

    def clean(self, strategies: Dict[str, Any] = None) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Execute all cleaning operations.
        
        Args:
            strategies: Dictionary of cleaning strategies and their parameters
        
        Returns:
            Tuple of (cleaned DataFrame, cleaning report)
        """
        if strategies is None:
            strategies = {
                'standardize_names': True,
                'remove_duplicates': True,
                'handle_missing': 'auto',
                'detect_outliers': 3.0,
                'convert_types': True,
                'remove_low_variance': 0.01
            }

        # Get critical columns from the strategies if provided
        critical_columns = strategies.get('critical_columns', [])

        # Always convert types first to ensure proper data handling
        if strategies.get('convert_types', True):
            self.convert_data_types()

        if strategies.get('standardize_names', True):
            self.standardize_column_names()
        
        if strategies.get('remove_duplicates', True):
            self.remove_duplicates()
        
        if 'handle_missing' in strategies:
            self.handle_missing_values(
                strategy=strategies['handle_missing'],
                critical_columns=critical_columns
            )
        
        if 'detect_outliers' in strategies:
            self.detect_outliers(strategies['detect_outliers'])
        
        if 'remove_low_variance' in strategies:
            self.remove_low_variance_columns(strategies['remove_low_variance'])

        # Reset index to ensure clean data
        self.df = self.df.reset_index(drop=True)
        
        return self.df, self.report 

class AIDataCleaner:
    """AI-powered data cleaning class that uses Gemini for cleaning decisions and validation."""
    
    def __init__(self, df: pd.DataFrame, llm: ChatGoogleGenerativeAI):
        """Initialize with a copy of the original dataframe and LLM instance."""
        self.original_df = df.copy()
        self.df = df.copy()
        self.llm = llm
        self.operations = AICleaningOperations(llm)
        self.cleaning_history = []
        self.validation_results = []
        self.report = CleaningReport(
            original_shape=df.shape,
            cleaned_shape=df.shape,
            missing_values_filled={},
            duplicates_removed=0,
            outliers_detected={},
            column_types_converted={},
            columns_dropped=[],
            standardized_columns=[],
            cleaning_steps=[],
            validation_results=[],
            ai_decisions=[],
            cleaning_context=AICleaningContext(
                column_patterns={},
                value_relationships={},
                detected_formats={},
                confidence_thresholds={},
                cleaning_history=[]
            )
        )

    async def _validate_cleaning_step(self, step_name: str, before_df: pd.DataFrame, after_df: pd.DataFrame) -> bool:
        """Use AI to validate each cleaning step."""
        prompt = f"""
        Validate the following data cleaning step: {step_name}
        
        Before cleaning metrics:
        - Shape: {before_df.shape}
        - Missing values: {before_df.isna().sum().sum()}
        - Duplicate rows: {before_df.duplicated().sum()}
        
        After cleaning metrics:
        - Shape: {after_df.shape}
        - Missing values: {after_df.isna().sum().sum()}
        - Duplicate rows: {after_df.duplicated().sum()}
        
        Please analyze if the cleaning step was successful and didn't introduce any issues.
        Consider:
        1. Data quality improvement
        2. Information preservation
        3. Critical column integrity
        4. Semantic consistency

        Respond with either VALID or INVALID, followed by a brief explanation.
        """
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        result = response.content.strip().upper().startswith("VALID")
        
        self.validation_results.append({
            "step": step_name,
            "valid": result,
            "explanation": response.content
        })
        
        return result

    async def _get_row_context(self, column: str, row_idx: int) -> Dict:
        """Get context information for a specific row and column."""
        row_data = self.df.loc[row_idx].to_dict()
        column_stats = {
            "type": str(self.df[column].dtype),
            "mean": float(self.df[column].mean()) if pd.api.types.is_numeric_dtype(self.df[column]) else None,
            "mode": self.df[column].mode().iloc[0] if len(self.df[column].mode()) > 0 else None,
            "unique_count": self.df[column].nunique()
        }
        
        # Find similar rows with non-null values
        pattern_examples = []
        if not pd.isna(row_data[column]):
            similar_rows = self.df[
                (self.df[column].notna()) & 
                (self.df.index != row_idx)
            ].head(5)
            pattern_examples = similar_rows.to_dict('records')
        
        return {
            "row_data": row_data,
            "column_stats": column_stats,
            "pattern_examples": pattern_examples
        }

    async def _find_suspicious_values(self, column: str) -> List[Dict]:
        """Identify suspicious values in a column."""
        suspicious = []
        
        # Get column statistics
        if pd.api.types.is_numeric_dtype(self.df[column]):
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find statistical outliers
            outliers = self.df[
                (self.df[column] < lower_bound) | 
                (self.df[column] > upper_bound)
            ]
            
            for idx, value in zip(outliers.index, outliers[column]):
                context = await self._get_row_context(column, idx)
                suspicious.append({
                    "value": value,
                    "context": context,
                    "pattern_violations": [
                        f"Value {value} is outside normal range [{lower_bound:.2f}, {upper_bound:.2f}]"
                    ],
                    "valid_examples": self.df[
                        (self.df[column] >= lower_bound) & 
                        (self.df[column] <= upper_bound)
                    ][column].sample(min(5, len(self.df))).tolist()
                })
        else:
            # For non-numeric columns, look for pattern violations
            value_counts = self.df[column].value_counts()
            rare_values = value_counts[value_counts == 1].index
            
            for value in rare_values:
                idx = self.df[self.df[column] == value].index[0]
                context = await self._get_row_context(column, idx)
                suspicious.append({
                    "value": value,
                    "context": context,
                    "pattern_violations": ["Unique/rare value in categorical column"],
                    "valid_examples": value_counts.nlargest(5).index.tolist()
                })
        
        return suspicious

    async def _find_similar_value_groups(self, column: str) -> List[Dict]:
        """Group similar values for standardization."""
        if self.df[column].dtype != 'object':
            return []
        
        value_counts = self.df[column].value_counts()
        groups = []
        processed_values = set()
        
        for value in value_counts.index:
            if value in processed_values:
                continue
                
            # Find similar values using string similarity
            similar_values = [
                v for v in value_counts.index 
                if v not in processed_values and self._string_similarity(str(value), str(v)) > 0.8
            ]
            
            if len(similar_values) > 1:
                groups.append({
                    "values": similar_values,
                    "frequency": {str(v): int(value_counts[v]) for v in similar_values},
                    "pattern": self._detect_pattern(similar_values)
                })
                processed_values.update(similar_values)
        
        return groups

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity ratio."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def _detect_pattern(self, values: List[str]) -> str:
        """Detect common pattern in a list of values."""
        if not values:
            return ""
            
        # Convert values to lowercase for pattern detection
        values = [str(v).lower() for v in values]
        
        # Check for common prefixes
        prefix = os.path.commonprefix(values)
        if prefix and len(prefix) > 1:
            return f"Common prefix: {prefix}"
            
        # Check for common suffixes
        suffix = os.path.commonprefix([v[::-1] for v in values])[::-1]
        if suffix and len(suffix) > 1:
            return f"Common suffix: {suffix}"
            
        # Check for numeric patterns
        numeric_parts = [re.findall(r'\d+', v) for v in values]
        if all(numeric_parts):
            return "Contains numbers in similar positions"
            
        return "Similar string length and character composition"

    async def clean(self, assessment: DataAssessment) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Main cleaning pipeline using AI operations.
        
        Steps:
        1. Type inference and conversion
        2. Value standardization
        3. Anomaly detection and correction
        4. Missing value imputation
        """
        try:
            # Store initial state
            self.cleaning_history.append(('initial', self.df.copy()))
            
            # Process each column that's not marked as critical
            for column in self.df.columns:
                if column in assessment.critical_columns:
                    continue
                
                # Store column state
                column_backup = self.df[column].copy()
                
                try:
                    # 1. Type inference and conversion
                    sample_data = {
                        "current_type": str(self.df[column].dtype),
                        "values": self.df[column].dropna().sample(min(100, len(self.df))).tolist(),
                        "semantic_hints": {
                            "name": column,
                            "classification": assessment.column_classifications.get(column, "unknown")
                        }
                    }
                    
                    inferred_type, conversion_func = await self.operations.ai_infer_and_convert_type(
                        column, sample_data
                    )
                    
                    if conversion_func:
                        self.df[column] = self.df[column].apply(conversion_func)
                        self.report.column_types_converted[column] = (
                            str(column_backup.dtype),
                            str(self.df[column].dtype)
                        )
                    
                    # 2. Value standardization
                    value_groups = await self._find_similar_value_groups(column)
                    if value_groups:
                        standardizations = await self.operations.ai_standardize_values(column, value_groups)
                        if standardizations:
                            self.df[column] = self.df[column].replace(standardizations)
                            self.report.standardized_columns.append(column)
                    
                    # 3. Anomaly detection and correction
                    suspicious_values = await self._find_suspicious_values(column)
                    if suspicious_values:
                        corrections = await self.operations.ai_correct_anomalies(column, suspicious_values)
                        if corrections:
                            # Handle both corrections and removals
                            for orig_value, new_value in corrections.items():
                                if new_value is None:  # Remove the row
                                    self.df = self.df[self.df[column] != orig_value]
                                else:  # Replace the value
                                    self.df[column] = self.df[column].replace(orig_value, new_value)
                            
                            self.report.outliers_detected[column] = [
                                i for i, v in enumerate(column_backup) 
                                if v in corrections
                            ]
                    
                    # 4. Missing value imputation
                    missing_mask = self.df[column].isna()
                    if missing_mask.any():
                        for idx in self.df[missing_mask].index:
                            row_context = await self._get_row_context(column, idx)
                            imputed_value = await self.operations.ai_impute_value(column, row_context)
                            
                            if imputed_value is not None:
                                self.df.at[idx, column] = imputed_value
                                self.report.missing_values_filled[column] = (
                                    self.report.missing_values_filled.get(column, 0) + 1
                                )
                    
                    # Validate column changes
                    is_valid = await self._validate_cleaning_step(
                        f"clean_{column}",
                        pd.DataFrame({column: column_backup}),
                        pd.DataFrame({column: self.df[column]})
                    )
                    
                    if not is_valid:
                        self.df[column] = column_backup
                        continue
                    
                except Exception as e:
                    print(f"Error cleaning column {column}: {str(e)}")
                    self.df[column] = column_backup
            
            # Update report
            self.report.cleaned_shape = self.df.shape
            
            # Final validation
            is_valid = await self._validate_cleaning_step(
                "final_validation",
                self.original_df,
                self.df
            )
            
            if not is_valid:
                self.df = self.original_df.copy()
                return self.df, self.report
            
            return self.df, self.report
            
        except Exception as e:
            print(f"Error in cleaning pipeline: {str(e)}")
            self.df = self.original_df.copy()
            return self.df, self.report 