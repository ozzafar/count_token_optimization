import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

num_of_samples_per_questions = 13
num_of_responders = 22
total_answers_per_question = num_of_samples_per_questions * num_of_responders
Q1_correct = 252
Q1_incorrect = total_answers_per_question - Q1_correct
Q2_correct = 176
Q2_the_same = 52
Q2_incorrect = total_answers_per_question - Q2_correct - Q2_the_same

# Pie chart labels
labels = ['Ours', 'SD']

# Data for pie chart
data = [Q1_correct, Q1_incorrect]

# Plotting the pie chart
fig, ax = plt.subplots()
wedges, _, _ = ax.pie(data, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])

ax.legend(wedges, labels, title="Responses", loc="best")

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the plot
plt.savefig('human_study_q1.pdf', dpi=300, bbox_inches='tight')
plt.show()


# Pie chart labels
labels = ['Ours', 'SD', 'The same']

# Data for pie chart
data = [Q2_correct, Q2_the_same, Q2_incorrect]

# Plotting the pie chart
fig, ax = plt.subplots()
wedges, _, _ = ax.pie(data, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999', '#c2c2c2'])

ax.legend(wedges, labels, title="Responses", loc="best")

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the plot
plt.savefig('human_study_q2.pdf', dpi=300, bbox_inches='tight')
plt.show()