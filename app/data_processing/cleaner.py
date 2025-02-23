import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import datetime

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
            standardized_columns=[]
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