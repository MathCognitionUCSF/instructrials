import pandas as pd
import random

# Set the seed for reproducibility
random.seed(0)

# Function to generate random data for one child
def generate_data(sample_size=100):
    data = []
    for _ in range(sample_size):
        pre_test = random.randint(0, 100)
        # Assuming some improvement in post_test
        post_test = pre_test + random.randint(0, 20)
        post_test = min(post_test, 100)  # Ensure it doesn't exceed 100
        phonology_task1 = random.randint(0, 100)
        math_task_1 = random.randint(0, 100)
        age = random.randint(8, 12)
        gender = random.choice(['Male', 'Female'])
        data.append([pre_test, post_test, phonology_task1, math_task_1, age, gender])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['pre_test', 'post_test', 'phonology_task_1', 'math_task_1', 'age', 'gender'])
    df['pidn'] = range(1, len(df) + 1)

    return df
