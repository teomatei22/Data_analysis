import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Students_Grading_Dataset.csv")

numeric_cols = [
    "Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
    "Quizzes_Avg", "Participation_Score", "Projects_Score", "Total_Score",
    "Study_Hours_per_Week", "Stress_Level (1-10)", "Sleep_Hours_per_Night"
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

##################################
# Pie Chart of gender distribution
##################################
gender_counts = df["Gender"].value_counts()

fig, ax = plt.subplots()
ax.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%", startangle=90, colors=["skyblue", "pink", "lightgreen"])
ax.set_title("Gender Distribution")
plt.show()

#####################################
# Pie Chart of parent education level
#####################################
parent_edu_counts = df["Parent_Education_Level"].value_counts()

fig, ax = plt.subplots()
ax.pie(parent_edu_counts, labels=parent_edu_counts.index, autopct="%1.1f%%", startangle=90, colors=["lightblue", "lightcoral", "lightgreen", "gold"])
ax.set_title("Parent Education Level Distribution")
plt.show()


##########################################
# Density Plot of Final score distribution
##########################################
fig, ax = plt.subplots()
df["Final_Score"].plot(kind="density", ax=ax, color="purple")

ax.set_title("Density Plot of Final Scores")
ax.set_xlabel("Final Score")
ax.set_ylabel("Density")
plt.show()


#################################################################
# Multi-Bar Histograms of average Midterm vs Final by departament
#################################################################
dept_avg = df.groupby("Department")[["Midterm_Score", "Final_Score"]].mean()

fig, ax = plt.subplots()
x = np.arange(len(dept_avg))  # Positions for bars
width = 0.35

ax.bar(x - width/2, dept_avg["Midterm_Score"], width, label="Midterm", color="blue")
ax.bar(x + width/2, dept_avg["Final_Score"], width, label="Final", color="orange")

ax.set_xticks(x)
ax.set_xticklabels(dept_avg.index, rotation=45, ha="right")
ax.set_ylabel("Average Score")
ax.set_title("Avg Midterm vs. Final by Department")
ax.legend()
plt.show()

############################################################
# Multi-Bar Histograms of average Midterm vs Final by gender
############################################################
gender_avg = df.groupby("Gender")[["Midterm_Score", "Final_Score"]].mean()

fig, ax = plt.subplots()
x = np.arange(len(gender_avg))  # Positions for bars
width = 0.35

ax.bar(x - width/2, gender_avg["Midterm_Score"], width, label="Midterm", color="blue")
ax.bar(x + width/2, gender_avg["Final_Score"], width, label="Final", color="orange")

ax.set_xticks(x)
ax.set_xticklabels(gender_avg.index)
ax.set_ylabel("Average Score")
ax.set_title("Avg Midterm vs. Final by Gender")
ax.legend()
plt.show()

#######################################################
# Box-and-Wiskers of study hours distribution by gender
#######################################################
fig, ax = plt.subplots()

genders = df["Gender"].unique()
study_hours_by_gender = [df.loc[df["Gender"] == gender, "Study_Hours_per_Week"] for gender in genders]

ax.boxplot(study_hours_by_gender, labels=genders, patch_artist=True)

ax.set_title("Study Hours Distribution by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Study Hours per Week")
plt.show()
