import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from transformers import AutoTokenizer

# ================= CONFIGURATION =================
# 请确保路径正确
JSON_FILE_PATH = '/inspire/hdd/project/qproject-fundationmodel/xiashijie-240108120112/sjxia/agentnote/eval/FoldAgent/output/BrowseComp-Plus/minimax-2.1/reactagent/128k/results_20260109_145013.json'
MODEL_ID = "/inspire/hdd/project/qproject-fundationmodel/xiashijie-240108120112/sjxia/models/minimax-m2.1"
LENGTH_THRESHOLD = 127 * 1024  # Threshold for "Long" vs "Short" failures
# ===============================================

def search_tool():
    search = {
        'type': 'function',
        'function': {
            "name": "search",
            "description": "Performs a web search: supply a string 'query' and optional 'topk'. The tool retrieves the top 'topk' results (default 10) for the query, returning their docid, url, and document content (may be truncated based on token limits).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query string for the search."
                    },
                    "topk": {
                        "type": "integer",
                        "description": "Return the top k pages.",
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
    }
    open_page = {
        'type': 'function',
        'function': {
            'name': 'open_page',
            'description': (
                "Open a page by docid or URL and return the complete content. "
                "Provide either 'docid' or 'url'; if both are provided, prefer 'docid'. "
                "The docid or URL must come from prior search tool results."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'docid': {
                        'type': 'string',
                        'description': 'Document ID from search results to resolve and fetch.',
                    },
                    'url': {
                        'type': 'string',
                        'description': 'Absolute URL from search results to fetch.',
                    },
                },
                'required': [],
            },
        },
    }
    finish = {
        'type': 'function',
        'function': {
            'name': 'finish',
            'description': """Return the final result when you have a definitive answer or cannot progress further. Provide a concise answer plus a brief, evidence-grounded explanation.""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'answer': {
                        'type': 'string',
                        'description': 'A succinct, final answer.',
                    },
                    'explanation': {
                        'type': 'string',
                        'description': 'A brief explanation for your final answer. For this section only, cite evidence documents inline by placing their docids in square brackets at the end of sentences (e.g., [20]). Do not include citations anywhere else.',
                    },
                    'confidence': {
                        'type': 'string',
                        'description': 'Confidence: your confidence score between 0% and 100% for your answer',
                    },
                },
                'required': ['answer', 'explanation'],
            },
        },
    }
    return [search, open_page, finish]

def get_category(score, predicted_answer, token_length):
    """
    Classify the result into 3 categories.
    """
    has_answer = predicted_answer is not None and str(predicted_answer).strip() != ""

    if score == 1:
        return "1. Correct Answer"

    elif score == 0:
        if has_answer:
            return "2. Incorrect Answer"
        elif token_length < LENGTH_THRESHOLD:
            return "3. No Answer (within)"
        elif token_length >= LENGTH_THRESHOLD:
            return "4. No Answer (over)"

    return "Unknown/Other"

def normalize_tool_calls(messages):
    """
    Normalize tool call arguments to dicts for tokenizer.apply_chat_template.
    """
    processed_messages = []
    for msg in messages:
        new_msg = msg.copy()

        tool_calls = new_msg.get('tool_calls')
        if new_msg.get('role') == 'assistant' and tool_calls:
            new_tool_calls = []
            for tool_call in tool_calls:
                new_tc = tool_call.copy()
                if 'function' in new_tc:
                    new_func = new_tc['function'].copy()
                    args = new_func.get('arguments')

                    if isinstance(args, str):
                        try:
                            new_func['arguments'] = json.loads(args)
                        except json.JSONDecodeError:
                            print(f"Warning: Failed to parse arguments JSON: {args}")
                            pass

                    new_tc['function'] = new_func
                new_tool_calls.append(new_tc)
            new_msg['tool_calls'] = new_tool_calls

        processed_messages.append(new_msg)

    return processed_messages

def split_agent_sessions(messages):
    if messages:
        return [messages]
    return []

def print_stats_table(df_subset, title_prefix="ANALYSIS"):
    """
    Helper function to aggregate stats and print the table for a given DataFrame subset.
    """
    if df_subset.empty:
        print(f"\n[!] Warning: No data found for {title_prefix}")
        return

    total_count = len(df_subset)
    table_width = 120

    # Aggregation
    stats = df_subset.groupby('category').agg(
        count=('instance_id', 'count'),
        peak_max=('peak_length', 'max'),
        peak_avg=('peak_length', 'mean'),
        avg_len_avg=('average_length', 'mean'),
        turn_max=('turn_number', 'max'),
        turn_avg=('turn_number', 'mean'),
        session_max=('session_number', 'max'),
        session_avg=('session_number', 'mean')
    )

    stats['ratio (%)'] = (stats['count'] / total_count) * 100
    stats = stats.sort_index()

    # Max Table
    print("\n" + "=" * table_width)
    print(f"📊 {title_prefix} (MAX) (Samples: {total_count})")
    print("=" * table_width)
    print("-" * table_width)
    print(f"{'Category':<22} | {'Ratio':<7} | {'PeakMax(K)':<10} | {'TurnMax':<7} | {'SessMax':<7}")
    print("-" * table_width)

    for category, row in stats.iterrows():
        ratio_str = f"{row['ratio (%)']:.1f}%"
        peak_max_str = f"{row['peak_max'] / 1024:.1f}K"
        turn_max_str = f"{row['turn_max']:.0f}"
        sess_max_str = f"{row['session_max']:.0f}"
        print(
            f"{category:<22} | {ratio_str:<7} | {peak_max_str:<10} | {turn_max_str:<7} | {sess_max_str:<7}"
        )

    print("-" * table_width)

    # Avg Table
    print("\n" + "=" * table_width)
    print(f"📊 {title_prefix} (AVG) (Samples: {total_count})")
    print("=" * table_width)
    print("-" * table_width)
    print(f"{'Category':<22} | {'Ratio':<7} | {'PeakAvg(K)':<10} | {'AvgLen(K)':<10} | {'TurnAvg':<7} | {'SessAvg':<7}")
    print("-" * table_width)

    for category, row in stats.iterrows():
        ratio_str = f"{row['ratio (%)']:.1f}%"
        peak_avg_str = f"{row['peak_avg'] / 1024:.1f}K"
        avg_len_str = f"{row['avg_len_avg'] / 1024:.1f}K"
        turn_avg_str = f"{row['turn_avg']:.1f}"
        sess_avg_str = f"{row['session_avg']:.1f}"
        print(
            f"{category:<22} | {ratio_str:<7} | {peak_avg_str:<10} | {avg_len_str:<10} | {turn_avg_str:<7} | {sess_avg_str:<7}"
        )

    print("-" * table_width)

def analyze_performance(file_path, model_id):
    print(f"[*] Loading Tokenizer: {model_id} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"[!] Failed to load tokenizer.\nError: {e}")
        return

    print(f"[*] Reading JSON file: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    results = data.get('results', [])
    records = []
    tools = search_tool()

    print(f"[*] Processing {len(results)} records...")

    def process_item(item):
        score = item.get('score', 0)
        messages = item.get('messages', [])
        pred_ans = item.get('predicted_answer', None)
        instance_id = item.get('instance_id', 'unknown')
        data_source = item.get('data_source', 'unknown')

        sessions = split_agent_sessions(messages)
        session_lengths = []
        for session in sessions:
            processed_messages = normalize_tool_calls(session)
            token_ids = tokenizer.apply_chat_template(
                processed_messages,
                tokenize=True,
                add_generation_prompt=False,
                tools=tools
            )
            session_lengths.append(len(token_ids))

        peak_length = max(session_lengths) if session_lengths else 0
        average_length = sum(session_lengths) / len(session_lengths) if session_lengths else 0
        session_number = len(sessions)
        turn_number = sum(
            1
            for msg in messages
            if msg.get('role') == 'assistant'
            and not (msg.get('content', None) == "" and not msg.get('tool_calls'))
        )
        category = get_category(score, pred_ans, peak_length)

        return {
            'instance_id': instance_id,
            'data_source': data_source,
            'category': category,
            'peak_length': peak_length,
            'average_length': average_length,
            'turn_number': turn_number,
            'session_number': session_number,
            'score': score
        }

    if results:
        with ThreadPoolExecutor(max_workers=64) as executor:
            records = list(executor.map(process_item, results))

    # ================= STATISTICAL ANALYSIS =================
    df = pd.DataFrame(records)

    print(f"\nmax response length allowed {LENGTH_THRESHOLD}")

    # 1. Overall Results
    print_stats_table(df, "OVERALL PERFORMANCE (PEAK AGENT CONTEXT)")
    high_turn_df = df[df['turn_number'] > 190]
    if not high_turn_df.empty:
        print("\ninstance_id(s) with turn_number > 190:")
        for instance_id in high_turn_df['instance_id'].tolist():
            print(f" - {instance_id}")

    # 2. Results Grouped by Data Source
    unique_sources = df['data_source'].unique()

    if len(unique_sources) > 0:
        print("\n" + "#" * 40)
        print(" accuracy by DATA SOURCE ".center(40, "#"))
        print("#" * 40)

        acc_stats = (
            df.groupby('data_source')
            .agg(
                count=('instance_id', 'count'),
                acc=('score', lambda s: (s == 1).mean() * 100)
            )
            .reset_index()
            .sort_values('data_source')
        )

        print(f"{'Source':<30} | {'Count':<7} | {'Acc(%)':<7}")
        print("-" * 55)
        for _, row in acc_stats.iterrows():
            source = str(row['data_source'])
            count = int(row['count'])
            acc = row['acc']
            print(f"{source:<30} | {count:<7} | {acc:>6.1f}")

    print("\n" + "=" * 50)
    print(" 📊 FINAL GLOBAL STATISTICS ".center(50, "="))
    print("=" * 50)

    if not df.empty:
        total_samples = len(df)
        global_acc = (df['score'] == 1).mean() * 100
        global_peak_avg = df['peak_length'].mean()
        global_avg_len_avg = df['average_length'].mean()
        global_turn_avg = df['turn_number'].mean()
        global_sess_avg = df['session_number'].mean()
        print(f"{'Metric':<25} | {'Value'}")
        print("-" * 50)
        print(f"{'Total Samples':<25} | {total_samples}")
        print(f"{'Overall Accuracy':<25} | {global_acc:.2f}%")
        print(f"{'Avg Peak Length':<25} | {global_peak_avg / 1024:.2f}K")
        print(f"{'Avg Session Length':<25} | {global_avg_len_avg / 1024:.2f}K")
        print(f"{'Avg Turns':<25} | {global_turn_avg:.2f}")
        print(f"{'Avg Sessions':<25} | {global_sess_avg:.2f}")
        print("-" * 50)
    else:
        print("No data available.")

if __name__ == "__main__":
    analyze_performance(JSON_FILE_PATH, MODEL_ID)
