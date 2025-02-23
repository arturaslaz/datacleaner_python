import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv
from data_processing.cleaner import DataCleaner
from ai_agents.assessment_agent import DataAssessmentAgent
import asyncio

# Load environment variables from .env file if in development
if os.getenv('ENVIRONMENT') != 'production':
    load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Data Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'cleaning_report' not in st.session_state:
    st.session_state.cleaning_report = None
if 'assessment' not in st.session_state:
    st.session_state.assessment = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

def reset_state():
    """Reset all session state variables"""
    # Clear all session state variables except the file uploader
    keys_to_clear = [
        'uploaded_df',
        'cleaned_df',
        'cleaning_report',
        'assessment',
        'file_processed'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = None

    # Reset the file processed flag
    st.session_state.file_processed = False
    
    # Increment the reset counter to force file uploader to reset
    st.session_state.reset_counter = st.session_state.get('reset_counter', 0) + 1

async def analyze_and_clean_data(df: pd.DataFrame):
    """Analyze the dataset and perform cleaning operations."""
    # Initialize the assessment agent
    agent = DataAssessmentAgent()
    
    # Get AI assessment and recommendations
    assessment = await agent.assess_dataset(df)
    
    # Get cleaning strategy based on assessment
    strategy = agent.get_cleaning_strategy(assessment)
    
    # Initialize and run the cleaner
    cleaner = DataCleaner(df)
    cleaned_df, report = cleaner.clean(strategy)
    
    return cleaned_df, report, assessment

def main():
    # Check for required environment variables
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("""
        Missing GOOGLE_API_KEY environment variable. 
        Please set it in your .env file for local development 
        or in your deployment environment.
        """)
        st.stop()

    st.title("🧹 AI Data Cleaner")
    st.write("""
    Upload your CSV or Excel file and let AI help you clean your data.
    The app will analyze your data, suggest improvements, and perform cleaning automatically.
    """)

    # Add a button to start new cleaning process
    if st.button("Start New Cleaning Process"):
        reset_state()
        st.rerun()

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload your data file here. Supported formats: CSV, Excel (.xlsx, .xls)",
        key=f"file_uploader_{st.session_state.get('reset_counter', 0)}"
    )

    if uploaded_file is not None and not st.session_state.file_processed:
        try:
            # Create a progress bar
            progress_text = "Loading and analyzing your data..."
            progress_bar = st.progress(0, text=progress_text)

            # Read the file
            file_extension = Path(uploaded_file.name).suffix.lower()
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            else:  # Excel file
                df = pd.read_excel(uploaded_file)

            # Store the original dataframe in session state
            st.session_state.uploaded_df = df
            progress_bar.progress(30, text="Data loaded successfully!")

            # Display basic information about the dataset
            st.subheader("📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())

            # Display the raw data
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Display column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isna().sum(),
                'Unique Values': df.nunique()
            }).astype(str)  # Convert all values to strings to avoid Arrow serialization issues
            st.dataframe(col_info, use_container_width=True)

            progress_bar.progress(100, text="Analysis complete! Ready for cleaning.")
            st.session_state.file_processed = True

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.stop()

    # Add a button to trigger cleaning
    if st.session_state.uploaded_df is not None and st.button("🧹 Clean Data", type="primary"):
        with st.spinner("Analyzing and cleaning your data..."):
            # Run the async cleaning process
            cleaned_df, report, assessment = asyncio.run(analyze_and_clean_data(st.session_state.uploaded_df))
            
            # Store results in session state
            st.session_state.cleaned_df = cleaned_df
            st.session_state.cleaning_report = report
            st.session_state.assessment = assessment

    # Display cleaning results if available
    if st.session_state.cleaned_df is not None:
        st.success("Data cleaning completed successfully!")
        
        # Show AI assessment
        st.subheader("🤖 AI Assessment")
        st.write(f"Data Quality Score: {st.session_state.assessment.quality_score:.2f}")
        
        with st.expander("Issues Detected"):
            for issue in st.session_state.assessment.issues_detected:
                st.write(f"• {issue}")
        
        # Show cleaning report
        st.subheader("🧹 Cleaning Report")
        st.write(f"Original shape: {st.session_state.cleaning_report.original_shape}")
        st.write(f"Cleaned shape: {st.session_state.cleaning_report.cleaned_shape}")
        
        if st.session_state.cleaning_report.duplicates_removed > 0:
            st.write(f"Duplicates removed: {st.session_state.cleaning_report.duplicates_removed}")
        
        if st.session_state.cleaning_report.missing_values_filled:
            st.write("Missing values filled:")
            for col, count in st.session_state.cleaning_report.missing_values_filled.items():
                st.write(f"• {col}: {count} values")
        
        if st.session_state.cleaning_report.outliers_detected:
            st.write("Outliers detected:")
            for col, indices in st.session_state.cleaning_report.outliers_detected.items():
                st.write(f"• {col}: {len(indices)} outliers")
        
        if st.session_state.cleaning_report.column_types_converted:
            st.write("Column types converted:")
            for col, (old_type, new_type) in st.session_state.cleaning_report.column_types_converted.items():
                st.write(f"• {col}: {old_type} → {new_type}")

        # Display data comparison
        st.subheader("📊 Data Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Data")
            st.dataframe(
                st.session_state.uploaded_df.dropna(how='all'),
                use_container_width=True,
                height=400
            )
        
        with col2:
            st.write("Cleaned Data")
            st.dataframe(
                st.session_state.cleaned_df.dropna(how='all'),
                use_container_width=True,
                height=400
            )
        
        # Add download button for cleaned data
        st.download_button(
            label="📥 Download Cleaned Data",
            data=st.session_state.cleaned_df.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

    # Add information about the app
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses AI to help you clean your data automatically. 
        It can handle common data issues like:
        - Missing values
        - Duplicates
        - Outliers
        - Inconsistent formatting
        - Data type mismatches
        """)

        st.header("Instructions")
        st.write("""
        1. Upload your data file (CSV or Excel)
        2. Review the data overview and analysis
        3. Click 'Clean Data' to start the cleaning process
        4. Download your cleaned data
        """)

if __name__ == "__main__":
    main() 