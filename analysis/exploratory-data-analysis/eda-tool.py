import pandas as pd
from ydata_profiling import ProfileReport

# Load a sample of your data
df = pd.read_csv("path/to/your/dataset_chunk_01.csv")

# Generate the report
profile = ProfileReport(df, title="Rocket League EDA - Initial Profile")

# Save the report to an HTML file
profile.to_file("initial_eda_report.html")