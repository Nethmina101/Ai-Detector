import pandas as pd

df = pd.read_csv("AI_Human.csv", low_memory=False)

# Clean column names
df.columns = [c.strip() for c in df.columns]

print("Columns:", df.columns.tolist())
print(df.head())

# Drop totally empty / unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Try to find the label column automatically
possible_label_cols = [c for c in df.columns if "generate" in c.lower() or "label" in c.lower()]
print("Possible label columns:", possible_label_cols)

# If your label column is not literally named 'generated', set it here:
LABEL_COL = possible_label_cols[0] if possible_label_cols else None

if LABEL_COL is None:
    raise ValueError("Could not find label column. Print df.columns and set LABEL_COL manually.")

# Remove rows where label is not 0/1 (like the note row)
df = df[df[LABEL_COL].isin([0, 1])]

print("\nLabel distribution:")
print(df[LABEL_COL].value_counts())
