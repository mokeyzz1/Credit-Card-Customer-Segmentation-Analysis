# generate_synthetic_data.py

import random
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
np.random.seed(42)
random.seed(42)

def get_income_bracket(annual_income):
    if annual_income < 40000:
        return 'Low'
    elif annual_income <= 100000:
        return 'Medium'
    else:
        return 'High'

def get_education(age):
    if age >= 30:
        options = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
        weights = [0.27, 0.18, 0.35, 0.15, 0.05]
    elif age >= 26:
        options = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
        weights = [0.30, 0.20, 0.35, 0.13, 0.02]  # Fewer advanced degrees for younger people
    elif age >= 24:
        options = ['High School', 'Associate', 'Bachelor', 'Master']
        weights = [0.35, 0.25, 0.30, 0.10]  # Some Master's possible by 24-25
    elif age >= 22:
        options = ['High School', 'Associate', 'Bachelor']
        weights = [0.40, 0.30, 0.30]  # Bachelor's possible by 22
    elif age >= 20:
        options = ['High School', 'Associate']
        weights = [0.70, 0.30]  # Mostly HS, some Associate
    else:
        options = ['High School']
        weights = [1.0]  # Only HS for 18-19 year olds
    return random.choices(options, weights=weights)[0]

def get_realistic_income(age, education, employment_status):
    """Generate realistic income based on age, education, and employment"""
    # Base income by education level
    if education == 'High School':
        base_min, base_max = 25000, 55000
    elif education == 'Associate':
        base_min, base_max = 30000, 65000
    elif education == 'Bachelor':
        base_min, base_max = 40000, 85000
    elif education == 'Master':
        base_min, base_max = 55000, 120000
    elif education == 'Doctorate':
        base_min, base_max = 70000, 150000
    else:
        base_min, base_max = 25000, 55000
    
    # Adjust for employment status
    if employment_status == 'Student':
        # Students earn much less, mostly part-time/internship income
        return round(random.uniform(12000, 35000), 2)
    elif employment_status == 'Part-time':
        # Part-time but educated people should earn more per hour
        if education in ['Bachelor', 'Master', 'Doctorate']:
            # Educated part-time workers (consultants, professionals) 
            min_income = min(30000, base_min * 0.5)  # At least $30k for educated PT
            max_income = base_max * 0.7  # Up to 70% for highly skilled PT
            return round(random.uniform(min_income, max_income), 2)
        else:
            # Regular part-time is 30-60% of full-time
            return round(random.uniform(base_min * 0.3, base_min * 0.6), 2)
    elif employment_status == 'Unemployed':
        # Very low income (unemployment benefits, temp work)
        return round(random.uniform(8000, 25000), 2)
    elif employment_status == 'Retired':
        # Retirement income varies by education (better planning/savings)
        if education in ['Master', 'Doctorate']:
            # Higher education = better retirement planning
            retirement_income = random.uniform(base_min * 0.6, base_max * 0.9)
        elif education == 'Bachelor':
            # Good retirement planning
            retirement_income = random.uniform(base_min * 0.5, base_max * 0.8)
        else:
            # Basic retirement (more dependent on Social Security)
            retirement_income = random.uniform(base_min * 0.4, base_max * 0.7)
        return round(retirement_income, 2)
    elif employment_status == 'Full-time':
        # Full-time gets full range, adjusted for experience (age)
        experience_years = max(0, age - 22)  # Assume work starts around 22
        
        # Experience multiplier (caps at reasonable levels)
        if experience_years <= 5:
            exp_multiplier = 1.0 + (experience_years * 0.05)  # 5% per year early career
        elif experience_years <= 15:
            exp_multiplier = 1.25 + ((experience_years - 5) * 0.03)  # 3% per year mid career
        else:
            exp_multiplier = 1.55 + ((experience_years - 15) * 0.01)  # 1% per year senior
        
        # Cap the multiplier to avoid unrealistic salaries
        exp_multiplier = min(exp_multiplier, 2.5)
        
        income_min = base_min * exp_multiplier
        income_max = base_max * exp_multiplier
        
        return round(random.uniform(income_min, income_max), 2)
    elif employment_status == 'Self-employed':
        # Self-employed has wider variance
        variance_multiplier = random.uniform(0.7, 2.0)  # Can be lower or much higher
        experience_years = max(0, age - 25)  # Assume self-employment starts later
        exp_multiplier = 1.0 + (experience_years * 0.02)  # Slower growth
        
        income = (base_min + base_max) / 2 * variance_multiplier * exp_multiplier
        return round(income, 2)
    else:
        # Default fallback
        return round(random.uniform(base_min, base_max), 2)

def get_realistic_credit_score(age, income, employment_status):
    """Generate realistic credit score based on multiple factors"""
    # Base score by income
    if income < 30000:
        base_score = random.randint(450, 650)
    elif income < 50000:
        base_score = random.randint(550, 720)
    elif income < 100000:
        base_score = random.randint(650, 780)
    else:
        base_score = random.randint(700, 850)
    
    # Age adjustment (credit history length)
    if age < 23:
        age_adjustment = random.randint(-50, 0)  # Young people have shorter history
    elif age < 30:
        age_adjustment = random.randint(-20, 20)
    else:
        age_adjustment = random.randint(0, 30)  # Older people generally have better scores
    
    # Employment stability adjustment
    if employment_status == 'Unemployed':
        emp_adjustment = random.randint(-100, -30)
    elif employment_status == 'Student':
        emp_adjustment = random.randint(-30, 10)
    elif employment_status == 'Part-time':
        emp_adjustment = random.randint(-20, 10)
    elif employment_status == 'Full-time':
        emp_adjustment = random.randint(0, 30)
    elif employment_status == 'Self-employed':
        emp_adjustment = random.randint(-40, 40)  # More variable
    else:  # Retired
        emp_adjustment = random.randint(-10, 20)
    
    final_score = base_score + age_adjustment + emp_adjustment
    return max(300, min(850, final_score))  # Constrain to valid range

def get_improved_card_type(score, income):
    """Improved card type logic"""
    if score < 600:
        return 'Standard'
    elif income > 120000 and score > 750:
        return random.choices(['Gold', 'Platinum', 'Signature'], [0.3, 0.4, 0.3])[0]
    elif income > 80000 and score > 700:
        return random.choices(['Standard', 'Gold', 'Platinum'], [0.1, 0.6, 0.3])[0]
    elif income > 50000 and score > 650:
        return random.choices(['Standard', 'Gold'], [0.4, 0.6])[0]
    elif income > 100000:  # High income but lower score - still get better cards
        return random.choices(['Standard', 'Gold'], [0.3, 0.7])[0]
    else:
        return 'Standard'

def get_default_risk(credit_score):
    if credit_score >= 800:
        return round(random.uniform(0.01, 0.02), 3)
    elif credit_score >= 740:
        return round(random.uniform(0.02, 0.04), 3)
    elif credit_score >= 670:
        return round(random.uniform(0.05, 0.09), 3)
    elif credit_score >= 580:
        return round(random.uniform(0.10, 0.20), 3)
    else:
        return round(random.uniform(0.25, 0.35), 3)

def get_card_type(score, income):
    if score < 600:
        return 'Standard'
    elif income > 150000 and score > 750:
        return random.choices(['Gold', 'Platinum', 'Signature'], [0.4, 0.4, 0.2])[0]
    elif income > 80000 and score > 700:
        return random.choices(['Standard', 'Gold', 'Platinum'], [0.2, 0.5, 0.3])[0]
    elif income > 50000 and score > 650:
        return random.choices(['Standard', 'Gold'], [0.6, 0.4])[0]
    else:
        return 'Standard'

def get_credit_limit(income, score):
    if income < 40000:
        return round(random.uniform(300, 3000), 2)
    elif income <= 100000:
        return round(random.uniform(3000, 15000 if score >= 650 else 7000), 2)
    else:
        return round(random.uniform(15000, 50000 if score >= 650 else 15000), 2)

def get_utilization(payer):
    if payer == 'Transactor':
        return round(random.uniform(0.05, 0.2), 2)
    elif payer == 'Revolver':
        return round(random.uniform(0.3, 0.7), 2)
    else:
        return round(random.uniform(0.4, 0.9), 2)

def get_apr(score):
    if score < 600:
        return round(random.uniform(23, 29), 2)
    elif score < 700:
        return round(random.uniform(18, 25), 2)
    elif score < 760:
        return round(random.uniform(15, 22), 2)
    else:
        return round(random.uniform(12, 18), 2)

def get_payer_type():
    return random.choices(['Transactor', 'Revolver', 'Minimum Payer'], [0.36, 0.40, 0.24])[0]

def get_state():
    """Generate realistic state distribution"""
    states = {
        'CA': 0.12, 'TX': 0.09, 'FL': 0.065, 'NY': 0.06, 'PA': 0.04,
        'IL': 0.04, 'OH': 0.035, 'GA': 0.032, 'NC': 0.032, 'MI': 0.03,
        'Other': 0.393  # Remaining 40 states
    }
    state = random.choices(list(states.keys()), weights=list(states.values()))[0]
    return state

def get_employment_status(age, education, income=50000):
    """Generate employment status based on age, education, and income potential"""
    if age < 22:
        # Young people - education matters a lot, but also income potential
        if education in ['Bachelor', 'Master', 'Doctorate']:
            # Highly educated young people are likely students
            return random.choices(['Student', 'Part-time', 'Full-time'], [0.7, 0.2, 0.1])[0]
        elif education == 'Associate':
            # Some college - mix of students and early workers
            return random.choices(['Student', 'Part-time', 'Full-time'], [0.5, 0.3, 0.2])[0]
        else:  # High School
            # High school graduates - income potential affects employment choice
            if income > 60000:  # High earning potential
                return random.choices(['Full-time', 'Part-time', 'Student'], [0.5, 0.3, 0.2])[0]
            else:
                return random.choices(['Student', 'Part-time', 'Full-time'], [0.3, 0.4, 0.3])[0]
    elif age > 65:
        return random.choices(['Retired', 'Part-time', 'Full-time'], [0.7, 0.2, 0.1])[0]
    else:
        return random.choices(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], [0.8, 0.1, 0.08, 0.02])[0]

def get_dependents(age, marital_status):
    """Generate dependents information with age-appropriate logic"""
    # Very young people (18-21) should rarely have dependents
    if age <= 21:
        if marital_status == 'Single':
            return random.choices([0, 1], [0.98, 0.02])[0]  # 98% no dependents
        elif marital_status == 'Married':
            return random.choices([0, 1], [0.85, 0.15])[0]  # Even married young people often wait
        else:  # Divorced (very rare at this age)
            return random.choices([0, 1], [0.70, 0.30])[0]
    
    # Young adults (22-25) - some start having children
    elif age <= 25:
        if marital_status == 'Single':
            return random.choices([0, 1], [0.90, 0.10])[0]
        elif marital_status == 'Married':
            return random.choices([0, 1, 2], [0.60, 0.30, 0.10])[0]
        else:  # Divorced
            return random.choices([0, 1], [0.60, 0.40])[0]
    
    # Prime family years (26-35)
    elif age <= 35:
        if marital_status == 'Single':
            return random.choices([0, 1, 2], [0.80, 0.15, 0.05])[0]
        elif marital_status == 'Married':
            return random.choices([0, 1, 2, 3], [0.25, 0.30, 0.35, 0.10])[0]
        else:  # Divorced
            return random.choices([0, 1, 2], [0.40, 0.40, 0.20])[0]
    
    # Established families (36-45)
    elif age <= 45:
        if marital_status == 'Single':
            return random.choices([0, 1, 2], [0.75, 0.20, 0.05])[0]
        elif marital_status == 'Married':
            return random.choices([0, 1, 2, 3, 4], [0.20, 0.25, 0.35, 0.15, 0.05])[0]
        else:  # Divorced
            return random.choices([0, 1, 2, 3], [0.30, 0.35, 0.25, 0.10])[0]
    
    # Older adults (45+) - kids may be grown
    else:
        if marital_status == 'Single':
            return random.choices([0, 1, 2], [0.80, 0.15, 0.05])[0]
        elif marital_status == 'Married':
            return random.choices([0, 1, 2, 3, 4], [0.40, 0.20, 0.25, 0.10, 0.05])[0]  # Many kids grown up
        else:  # Divorced
            return random.choices([0, 1, 2, 3], [0.45, 0.30, 0.20, 0.05])[0]

def get_marital_status(age):
    """Generate age-appropriate marital status"""
    if age < 22:
        # Very young people - almost all single
        return random.choices(['Single', 'Married'], [0.96, 0.04])[0]  # Only 4% married
    elif age < 25:
        # Early twenties - some start getting married, very rare divorces
        return random.choices(['Single', 'Married', 'Divorced'], [0.78, 0.21, 0.01])[0]
    elif age < 30:
        # Late twenties - more married
        return random.choices(['Single', 'Married', 'Divorced'], [0.45, 0.50, 0.05])[0]
    elif age < 40:
        # Thirties - peak marriage years
        return random.choices(['Single', 'Married', 'Divorced'], [0.25, 0.65, 0.10])[0]
    elif age < 50:
        # Forties - some divorces
        return random.choices(['Single', 'Married', 'Divorced'], [0.20, 0.60, 0.20])[0]
    else:
        # 50+ - more divorces, some widowed
        return random.choices(['Single', 'Married', 'Divorced'], [0.18, 0.55, 0.27])[0]

def generate_customer():
    age = random.randint(18, 85)
    gender = random.choice(['Male', 'Female'])
    marital_status = get_marital_status(age)  # Use age-appropriate marital status
    education = get_education(age)
    
    # Enhanced features - generate income first, then employment to avoid edge cases
    state = get_state()
    
    # Step 1: Generate initial income based on age and education only
    if education == 'High School':
        base_min, base_max = 25000, 55000
    elif education == 'Associate':
        base_min, base_max = 30000, 65000
    elif education == 'Bachelor':
        base_min, base_max = 40000, 85000
    elif education == 'Master':
        base_min, base_max = 55000, 120000
    elif education == 'Doctorate':
        base_min, base_max = 70000, 150000
    else:
        base_min, base_max = 25000, 55000
    
    # Apply age/experience multiplier
    experience_years = max(0, age - 22)
    if experience_years <= 5:
        exp_multiplier = 1.0 + (experience_years * 0.05)
    elif experience_years <= 15:
        exp_multiplier = 1.25 + ((experience_years - 5) * 0.03)
    else:
        exp_multiplier = 1.55 + ((experience_years - 15) * 0.01)
    exp_multiplier = min(exp_multiplier, 2.5)
    
    initial_income = round(random.uniform(base_min * exp_multiplier, base_max * exp_multiplier), 2)
    
    # Step 2: Generate employment status based on income potential
    employment_status = get_employment_status(age, education, initial_income)
    
    # Step 3: Adjust income based on employment reality
    income = get_realistic_income(age, education, employment_status)
    income_bracket = get_income_bracket(income)
    
    # Generate realistic credit score
    credit_score = get_realistic_credit_score(age, income, employment_status)
    default_risk = get_default_risk(credit_score)
    
    dependents = get_dependents(age, marital_status)
    
    # Tenure should be realistic based on age (assuming people get first card at 18)
    max_possible_tenure = max(1, (age - 18) * 12)  # At least 1 month for anyone 18+
    if age == 18:
        tenure = random.randint(1, 6)  # 18-year-olds: 1-6 months max
    else:
        tenure = random.randint(3, min(max_possible_tenure, 240))
    
    payer_type = get_payer_type()
    card_type = get_improved_card_type(credit_score, income)  # Use improved card type logic
    credit_limit = get_credit_limit(income, credit_score)
    utilization = get_utilization(payer_type)
    avg_spend = round(utilization * credit_limit, 2)
    apr = get_apr(credit_score)
    interest_paid = round((credit_limit * utilization * (apr / 100)) / 12, 2) if payer_type != 'Transactor' else 0.0
    pay_behavior = round({
        'Transactor': random.uniform(0.95, 1),
        'Revolver': random.uniform(0.80, 0.95),
        'Minimum Payer': random.uniform(0.5, 0.85)
    }[payer_type], 2)
    late_payments = min(12, int(round((1 - pay_behavior) * 12 + np.random.poisson(1))))
    annual_fee = round(random.choice([0, 0, 0, 95, 125, 199]), 2)
    rewards_earned = round(avg_spend * random.uniform(0.01, 0.02), 2)
    rewards_redeemed = round(rewards_earned * random.uniform(0.6, 1.0), 2)
    
    # More realistic profit calculation
    monthly_interest = interest_paid
    monthly_fee_revenue = annual_fee / 12
    monthly_rewards_cost = rewards_earned  # Full cost of rewards to bank
    monthly_default_cost = (default_risk * avg_spend * 12) / 12 * 0.1  # Annual default risk converted to monthly
    monthly_operational_cost = 2.0  # Fixed operational cost per customer
    
    profit = round(monthly_interest + monthly_fee_revenue - monthly_rewards_cost - monthly_default_cost - monthly_operational_cost, 2)

    return [
        fake.unique.bothify(text='CUST#####'),
        age, gender, marital_status, education, income, income_bracket, credit_score,
        default_risk, tenure, card_type, credit_limit, avg_spend, payer_type,
        interest_paid, pay_behavior, late_payments, utilization, apr,
        annual_fee, rewards_earned, rewards_redeemed, profit,
        # Only the 3 most important enhanced features
        state, employment_status, dependents
    ]

columns = [
    'Customer_ID', 'Age', 'Gender', 'Marital_Status', 'Education_Level', 'Annual_Income',
    'Income_Bracket', 'Credit_Score', 'Default_Risk_Score', 'Tenure_Months', 'Card_Type',
    'Credit_Limit', 'Avg_Monthly_Spend', 'Payer_Type', 'Interest_Paid', 'Payment_Behavior',
    'Late_Payments', 'Credit_Utilization', 'APR', 'Annual_Fee', 'Rewards_Earned',
    'Rewards_Redeemed', 'Profit_Contribution',
    # Only the 3 most important enhanced features
    'State', 'Employment_Status', 'Dependents'
]

data = [generate_customer() for _ in range(30000)]

df = pd.DataFrame(data, columns=columns)

# Add slight messiness (but not to critical financial calculations)
for col in ['Education_Level', 'Late_Payments']:  # Only apply to non-critical fields
    df.loc[df.sample(frac=0.01).index, col] = np.nan  # 1% missing values

def add_realistic_noise(df_input):
    """Add controlled realistic noise to specific fields without breaking business logic"""
    df_noisy = df_input.copy()
    
    # Use different seed for noise to keep it controllable but separate from main generation
    noise_state = np.random.RandomState(123)  # Fixed seed for reproducible noise
    
    # Credit Score noise (7% of customers get ±8-12 point variance)
    credit_score_candidates = df_noisy.sample(frac=0.07, random_state=noise_state)
    for idx in credit_score_candidates.index:
        noise = noise_state.randint(-12, 13)  # -12 to +12 points
        new_score = df_noisy.loc[idx, 'Credit_Score'] + noise
        df_noisy.loc[idx, 'Credit_Score'] = max(300, min(850, new_score))  # Keep in valid range
    
    # Late Payments noise (3% of good payers get 1 extra late payment)
    good_payers = df_noisy[df_noisy['Late_Payments'] <= 2]
    late_payment_candidates = good_payers.sample(frac=0.03, random_state=noise_state)
    for idx in late_payment_candidates.index:
        df_noisy.loc[idx, 'Late_Payments'] = min(12, df_noisy.loc[idx, 'Late_Payments'] + 1)
    
    # APR noise (2% get small rate adjustments)
    apr_candidates = df_noisy.sample(frac=0.02, random_state=noise_state)
    for idx in apr_candidates.index:
        noise = noise_state.uniform(-0.5, 0.5)  # ±0.5% rate variance
        new_apr = df_noisy.loc[idx, 'APR'] + noise
        df_noisy.loc[idx, 'APR'] = round(max(10.0, min(30.0, new_apr)), 2)  # Keep reasonable
    
    # Default Risk noise (4% get slight risk adjustments)
    risk_candidates = df_noisy.sample(frac=0.04, random_state=noise_state)
    for idx in risk_candidates.index:
        current_risk = df_noisy.loc[idx, 'Default_Risk_Score']
        noise_factor = noise_state.uniform(0.9, 1.1)  # ±10% variance
        new_risk = current_risk * noise_factor
        df_noisy.loc[idx, 'Default_Risk_Score'] = round(max(0.001, min(0.5, new_risk)), 3)
    
    # Utilization noise (4% get temporary spending spikes)
    util_candidates = df_noisy.sample(frac=0.04, random_state=noise_state)
    for idx in util_candidates.index:
        current_util = df_noisy.loc[idx, 'Credit_Utilization']
        if current_util < 0.8:  # Only spike if there's room
            spike = noise_state.uniform(0.05, 0.15)  # 5-15% spike
            new_util = min(0.95, current_util + spike)
            df_noisy.loc[idx, 'Credit_Utilization'] = round(new_util, 2)
            # Update related spending
            df_noisy.loc[idx, 'Avg_Monthly_Spend'] = round(new_util * df_noisy.loc[idx, 'Credit_Limit'], 2)
    
    return df_noisy

# Apply controlled noise (optional - comment out if you want original data)
df_with_noise = add_realistic_noise(df)

df_with_noise.to_csv('data/credit_card_data.csv', index=False)
print(f"✅ data/credit_card_data.csv saved with {len(df):,} rows and {len(df.columns)} features.")
print("📊 Features: 23 original + 3 enhanced (State, Employment_Status, Dependents)")