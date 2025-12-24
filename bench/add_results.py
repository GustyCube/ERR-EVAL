import json
from pathlib import Path

# Load each result file
results_dir = Path('results')
leaderboard_path = Path('c:/Users/Bennett/Desktop/MIRAGE/frontend/data/results.json')

with open(leaderboard_path) as f:
    leaderboard = json.load(f)

for result_file in results_dir.glob('*.json'):
    with open(result_file) as f:
        result = json.load(f)
    
    model_id = result['model_card']['model_id']
    print(f'Processing: {model_id}')
    
    # Calculate scores
    axis_scores = {
        'ambiguity_detection': [],
        'hallucination_avoidance': [],
        'localization_of_uncertainty': [],
        'response_strategy': [],
        'epistemic_tone': []
    }
    
    # Calculate cost and latency
    total_cost = 0
    total_latency = 0
    
    for item in result['item_results']:
        for axis, score_data in item['final_scores'].items():
            axis_scores[axis].append(score_data['score'])
        total_cost += item.get('cost', 0)
        total_latency += item.get('latency_ms', 0)
    
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0
    
    num_items = len(result['item_results'])
    avg_latency = total_latency / num_items if num_items > 0 else 0
    avg_cost = total_cost / num_items if num_items > 0 else 0
    
    track_scores = {}
    for track_summary in result.get('track_summaries', []):
        track_scores[track_summary['track']] = track_summary['mean_score']
    
    entry = {
        'rank': 0,  # Will be calculated after all entries loaded
        'model_id': model_id,
        'model_name': result['model_card']['model_name'],
        'provider': model_id.split('/')[0],
        'overall_score': result['overall_score'],
        'percentile': 50.0,  # Placeholder, can be calculated later
        'track_scores': track_scores,
        'axis_scores': {k: round(mean(v), 2) for k, v in axis_scores.items()},
        'items_evaluated': num_items,
        'avg_latency': round(avg_latency, 2),
        'avg_cost': round(avg_cost, 6),
        'evaluated_at': result['timestamp'],
    }
    
    # Remove existing entry for this model
    leaderboard['entries'] = [e for e in leaderboard['entries'] if e.get('model_id') != model_id]
    leaderboard['entries'].append(entry)
    print(f'  Added with score {result["overall_score"]}, cost ${avg_cost:.4f}/item, avg latency {avg_latency:.0f}ms')

# Sort entries by score and assign ranks
leaderboard['entries'].sort(key=lambda e: e['overall_score'], reverse=True)
for i, entry in enumerate(leaderboard['entries']):
    entry['rank'] = i + 1
    # Calculate percentile based on position
    entry['percentile'] = round(100 * (len(leaderboard['entries']) - i) / len(leaderboard['entries']), 1)

# Update providers
leaderboard['providers'] = result.get('providers', {})
leaderboard['generated_at'] = result['timestamp']

with open(leaderboard_path, 'w') as f:
    json.dump(leaderboard, f, indent=2)

print(f'\nUpdated {leaderboard_path} with {len(leaderboard["entries"])} entries')

