# --------------------------------------------------------------------------------------------------------------------
# Import Relevant Packages
# --------------------------------------------------------------------------------------------------------------------
# %%

from sklearn.model_selection import train_test_split
import pandas as pd
import os

#--------------------------------------------------------------------------------------------------------------------
# Set Working Directory
#--------------------------------------------------------------------------------------------------------------------
#%%

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, 'Data/')
print(f"Data Directory: {data_directory}")


#--------------------------------------------------------------------------------------------------------------------
# Load Data
#--------------------------------------------------------------------------------------------------------------------
#%%

tiny_stories_df = pd.read_csv(data_directory + 'Tiny Stories_Cleaned Dataset.csv')

#--------------------------------------------------------------------------------------------------------------------
# Data Splitting
#--------------------------------------------------------------------------------------------------------------------
#%%

# First split: 80% train, 20% temp (val+test)
train_df, rest_df = train_test_split(
    tiny_stories_df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Second split: Split temp into 50% val, 50% test (each 10% of total)
val_df, test_df = train_test_split(
    rest_df,
    test_size=0.5,
    random_state=42,
    shuffle=True
)


print(f"\nSplit sizes:")
print(f"Train: {len(train_df)} ({len(train_df)/len(tiny_stories_df)*100:.1f}%)")
print(f"Val:   {len(val_df)} ({len(val_df)/len(tiny_stories_df)*100:.1f}%)")
print(f"Test:  {len(test_df)} ({len(test_df)/len(tiny_stories_df)*100:.1f}%)")


#--------------------------------------------------------------------------------------------------------------------
# Save Splits
#--------------------------------------------------------------------------------------------------------------------
#%%

train_df.to_csv(data_directory + '/train.csv', index=False)
val_df.to_csv(data_directory + '/val.csv', index=False)
test_df.to_csv(data_directory + '/test.csv', index=False)

print("\nSplits saved successfully!")