import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset.csv')

# ------------------ COMMON STYLE ------------------
BG = '#0B1C2C'
AX_BG = '#1F3A5F'
TEXT = '#D1D5DB'

colors = ['#EF4444',  # red
          '#F59E0B',  # orange
          '#06B6D4',  # cyan
          '#7C3AED',  # purple
          '#10B981']  # green

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": AX_BG,
    "axes.edgecolor": "#94A3B8",
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT
})


# GRAPH 1: RISK FACTORS


hypertension = df[df['hypertension']==1]['stroke'].mean()*100
heart = df[df['heart_disease']==1]['stroke'].mean()*100
age60 = df[df['age']>60]['stroke'].mean()*100
glucose = df[df['avg_glucose_level']>140]['stroke'].mean()*100
norisk = df[(df['hypertension']==0)&(df['heart_disease']==0)]['stroke'].mean()*100

labels1 = ['Hypertension','Heart Disease','Age > 60','High Glucose','No Risk']
values1 = [round(hypertension), round(heart), round(age60), round(glucose), round(norisk)]

plt.figure(figsize=(7,8))
bars = plt.bar(labels1, values1, color=colors, edgecolor='white', linewidth=0.6)

# Value labels
for bar, val in zip(bars, values1):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val}', ha='center', fontsize=12, fontweight='bold')

plt.title('Stroke Rate by Risk Factor (%)', fontsize=16, fontweight='bold', color='#22D3EE')
plt.ylim(0, 10)
plt.ylabel('')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig('Risk_Factor.png', dpi=300)
plt.close()


# GRAPH 2: WORK TYPE

work_data = df.groupby('work_type')['stroke'].mean()*100
work_data = work_data.sort_values(ascending=False)

labels2 = work_data.index
values2 = work_data.values

plt.figure(figsize=(7,8))
bars = plt.bar(labels2, values2,
               color=[colors[3], colors[2], colors[1], colors[4], colors[0]],
               edgecolor='white', linewidth=0.6)

# Value labels
for bar, val in zip(bars, values2):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.title('Stroke Rate by Work Type (%)', fontsize=16, fontweight='bold')
plt.ylabel('Stroke Rate (%)')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig('Work_Type.png', dpi=300)
plt.close()