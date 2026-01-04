import json
import pandas as pd
import os

# Configuration
OUTPUT_DIR = r"d:\repos\ReasonTabQA\output"
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "Model_Accuracy_Report_All_Models.xlsx")

MODELS = [
    {
        "name": "Qwen3-32B",
        "file_pattern_en": "output_en_717_20percent_test_qwen3-32b.jsonl",
        "file_pattern_ch": "output_ch_1K_20percent_test_qwen3-32b.jsonl"
    },
    {
        "name": "Claude-Opus-4.5",
        "file_pattern_en": "output_en_717_20percent_test_sampled_claude-opus-4-5-20251101.jsonl",
        "file_pattern_ch": "output_ch_1K_20percent_test_sampled_claude-opus-4-5-20251101.jsonl"
    },
    {
        "name": "DeepSeek-V3.2",
        "file_pattern_en": "output_en_717_20percent_test_sampled_deepseek-v3.2.jsonl",
        "file_pattern_ch": "output_ch_1K_20percent_test_sampled_deepseek-v3.2.jsonl"
    },
    {
        "name": "DeepSeek-V3",
        "file_pattern_en": "output_en_717_20percent_test_sampled_deepseek-v3.jsonl",
        "file_pattern_ch": "output_ch_1K_20percent_test_sampled_deepseek-v3.jsonl"
    },
    {
        "name": "Gemini-3-Pro-Preview",
        "file_pattern_en": "output_en_717_20percent_test_sampled_gemini-3-pro-preview.jsonl",
        "file_pattern_ch": "output_ch_1K_20percent_test_sampled_gemini-3-pro-preview.jsonl"
    },
    {
        "name": "GPT-5.2",
        "file_pattern_en": "output_en_717_20percent_test_sampled_gpt-5.2.jsonl",
        "file_pattern_ch": "output_ch_1K_20percent_test_sampled_gpt-5.2.jsonl"
    }
]

def load_data(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Extract Difficulty
                t_diff = obj.get('table_difficulty', 'medium')
                q_diff = obj.get('question_difficulty', 'medium')
                
                # Extract Correctness
                is_correct = False
                if 'success@k' in obj:
                    val = obj['success@k']
                    if isinstance(val, bool): is_correct = val
                    else: is_correct = (val > 0)
                elif 'answer_correct' in obj:
                    is_correct = obj['answer_correct']
                    
                data.append({
                    't_diff': t_diff,
                    'q_diff': q_diff,
                    'correct': is_correct
                })
    else:
        print(f"Warning: File not found {file_path}")
        
    return pd.DataFrame(data)

def calculate_stats(df):
    if df.empty:
        return {}
    
    stats = {}
    
    # Overall
    stats['Overall'] = df['correct'].mean()
    
    # Question Difficulty
    q_stats = df.groupby('q_diff')['correct'].mean().to_dict()
    stats['Q_Easy'] = q_stats.get('easy', 0.0)
    stats['Q_Medium'] = q_stats.get('medium', 0.0)
    stats['Q_Hard'] = q_stats.get('hard', 0.0)
    
    # Table Difficulty
    t_stats = df.groupby('t_diff')['correct'].mean().to_dict()
    stats['T_Simple'] = t_stats.get('simple', t_stats.get('easy', 0.0))
    stats['T_Medium'] = t_stats.get('medium', 0.0)
    stats['T_Complex'] = t_stats.get('complex', t_stats.get('hard', 0.0))
    
    return stats

def main():
    rows_ch = []
    rows_en = []
    
    for model in MODELS:
        print(f"Processing {model['name']}...")
        
        # Chinese
        path_ch = os.path.join(OUTPUT_DIR, model['file_pattern_ch'])
        df_ch = load_data(path_ch)
        if not df_ch.empty:
            stats_ch = calculate_stats(df_ch)
            row_ch = {
                'Language': 'Chinese',
                'Model': model['name'],
                'Overall': stats_ch.get('Overall', 0),
                'Q_Easy': stats_ch.get('Q_Easy', 0),
                'Q_Medium': stats_ch.get('Q_Medium', 0),
                'Q_Hard': stats_ch.get('Q_Hard', 0),
                'T_Simple': stats_ch.get('T_Simple', 0),
                'T_Medium': stats_ch.get('T_Medium', 0),
                'T_Complex': stats_ch.get('T_Complex', 0)
            }
            rows_ch.append(row_ch)

        # English
        path_en = os.path.join(OUTPUT_DIR, model['file_pattern_en'])
        df_en = load_data(path_en)
        if not df_en.empty:
            stats_en = calculate_stats(df_en)
            row_en = {
                'Language': 'English',
                'Model': model['name'],
                'Overall': stats_en.get('Overall', 0),
                'Q_Easy': stats_en.get('Q_Easy', 0),
                'Q_Medium': stats_en.get('Q_Medium', 0),
                'Q_Hard': stats_en.get('Q_Hard', 0),
                'T_Simple': stats_en.get('T_Simple', 0),
                'T_Medium': stats_en.get('T_Medium', 0),
                'T_Complex': stats_en.get('T_Complex', 0)
            }
            rows_en.append(row_en)
    
    # Combine Chinese rows then English rows
    rows = rows_ch + rows_en

    if not rows:
        print("No data found to generate report.")
        return

    df_report = pd.DataFrame(rows)
    
    # Format columns order
    cols = ['Language', 'Model', 'Overall', 
            'Q_Easy', 'Q_Medium', 'Q_Hard', 
            'T_Simple', 'T_Medium', 'T_Complex']
    df_report = df_report[cols]
    
    # Rename columns
    df_report.columns = ['Language', 'Model', 'Overall Accuracy', 
                         'Question Easy', 'Question Medium', 'Question Hard',
                         'Table Simple', 'Table Medium', 'Table Complex']
    
    print("Generating Excel...")
    try:
        # Format as percentage strings
        df_report_fmt = df_report.copy()
        num_cols = df_report.select_dtypes(include=['float']).columns
        
        for col in num_cols:
            df_report_fmt[col] = df_report_fmt[col].apply(lambda x: f"{x * 100:.2f}%")
        
        # Save to Excel
        df_report_fmt.to_excel(OUTPUT_EXCEL, index=False)
        
        print(f"Report saved to {OUTPUT_EXCEL}")
        
    except Exception as e:
        print(f"Error saving Excel: {e}")

if __name__ == "__main__":
    main()
