import json
import os
from flask import Flask, request, render_template, jsonify
from rules_rag import RulesRAG

app = Flask(__name__)

# Initialize RAG system with processed SRD
SRD_PATH = os.environ.get('SRD_PATH', 'srd_processed.json')
rag = RulesRAG(SRD_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    data = request.json
    query_text = data.get('query', '')
    max_rules = int(data.get('max_rules', 10))
    
    if not query_text:
        return jsonify({'error': 'Query text is required'})
    
    # Process the query
    result = rag.query(query_text, max_rules=max_rules)
    
    # Return results as JSON
    return jsonify(result)

@app.route('/rule/<rule_id>')
def get_rule(rule_id):
    """Get detailed information about a specific rule."""
    rule = next((r for r in rag.rules if r['id'] == rule_id), None)
    
    if not rule:
        return jsonify({'error': 'Rule not found'})
    
    # Get related rules
    related_rules = []
    for edge in rag.graph_data['edges']:
        if edge['source'] == rule_id:
            target_rule = next((r for r in rag.rules if r['id'] == edge['target']), None)
            if target_rule:
                related_rules.append({
                    'id': target_rule['id'],
                    'title': target_rule['title'],
                    'relationship': edge['type']
                })
        elif edge['target'] == rule_id:
            source_rule = next((r for r in rag.rules if r['id'] == edge['source']), None)
            if source_rule:
                related_rules.append({
                    'id': source_rule['id'],
                    'title': source_rule['title'],
                    'relationship': f"is {edge['type']} by"
                })
    
    result = {
        'rule': rule,
        'related_rules': related_rules
    }
    
    return jsonify(result)

@app.route('/stats')
def get_stats():
    """Get statistics about the processed SRD."""
    rule_types = {}
    rule_scopes = {}
    rule_complexities = {}
    
    for rule in rag.rules:
        # Count rule types
        rule_type = rule.get('type', 'UNKNOWN')
        rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
        
        # Count rule scopes
        rule_scope = rule.get('scope', 'UNKNOWN')
        rule_scopes[rule_scope] = rule_scopes.get(rule_scope, 0) + 1
        
        # Count rule complexities
        rule_complexity = rule.get('complexity', 'UNKNOWN')
        rule_complexities[rule_complexity] = rule_complexities.get(rule_complexity, 0) + 1
    
    stats = {
        'total_rules': len(rag.rules),
        'total_terms': len(rag.terms),
        'total_relationships': len(rag.relationships),
        'rule_types': rule_types,
        'rule_scopes': rule_scopes, 
        'rule_complexities': rule_complexities
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RPG Rules RAG System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .sidebar {
            flex: 0 0 300px;
            padding: 20px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        .main-content {
            flex: 1;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .query-form {
            margin-bottom: 20px;
        }
        .query-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .btn {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .rule-card {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .rule-title {
            font-weight: bold;
            margin-top: 0;
            margin-bottom: 10px;
            color: #333;
        }
        .rule-meta {
            font-size: 0.8em;
            color: #666;
            margin-bottom: 10px;
        }
        .rule-content {
            margin-bottom: 10px;
        }
        .rule-references {
            font-size: 0.9em;
            color: #666;
        }
        .tag {
            display: inline-block;
            background-color: #e0e0e0;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-right: 5px;
        }
        .stats-container {
            margin-top: 30px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <h1>RPG Rules RAG System</h1>
    
    <div class="container">
        <div class="sidebar">
            <h2>Query Rules</h2>
            <div class="query-form">
                <input type="text" id="query-input" class="query-input" placeholder="Enter your rules question...">
                <div>
                    <label for="max-rules">Max rules:</label>
                    <input type="number" id="max-rules" value="5" min="1" max="20" style="width: 60px;">
                </div>
                <button class="btn" id="query-btn">Search Rules</button>
            </div>
            
            <div class="stats-container">
                <h3>System Stats</h3>
                <div id="stats-content">Loading stats...</div>
            </div>
        </div>
        
        <div class="main-content">
            <div id="loading" class="loading">
                Searching for relevant rules...
            </div>
            
            <div id="results-container">
                <p>Enter a rules question to see relevant rules from the SRD.</p>
            </div>
        </div>
    </div>
    
    <script>
        // Load stats when page loads
        window.addEventListener('DOMContentLoaded', loadStats);
        
        // Add event listeners
        document.getElementById('query-btn').addEventListener('click', handleQuery);
        document.getElementById('query-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                handleQuery();
            }
        });
        
        function handleQuery() {
            const queryText = document.getElementById('query-input').value.trim();
            const maxRules = document.getElementById('max-rules').value;
            
            if (!queryText) {
                alert('Please enter a query');
                return;
            }
            
            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results-container').innerHTML = '';
            
            // Send query to server
            fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: queryText,
                    max_rules: maxRules
                })
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data);
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('results-container').innerHTML = '<p>Error processing query</p>';
            })
            .finally(() => {
                document.getElementById('loading').style.display = 'none';
            });
        }
        
        function displayResults(data) {
            const container = document.getElementById('results-container');
            
            // Clear previous results
            container.innerHTML = '';
            
            // Display query info
            const queryInfo = document.createElement('div');
            queryInfo.innerHTML = `
                <h2>Results for: "${data.query}"</h2>
                <p>Found ${data.rule_count} relevant rules</p>
            `;
            container.appendChild(queryInfo);
            
            // Display detected features if any
            if (data.features && Object.values(data.features).some(v => Array.isArray(v) && v.length > 0)) {
                const featuresDiv = document.createElement('div');
                featuresDiv.className = 'rule-card';
                
                let featuresHtml = '<h3>Detected Query Features</h3>';
                
                if (data.features.detected_terms.length > 0) {
                    featuresHtml += `<p><strong>Terms:</strong> ${data.features.detected_terms.join(', ')}</p>`;
                }
                
                if (data.features.detected_actions.length > 0) {
                    featuresHtml += `<p><strong>Actions:</strong> ${data.features.detected_actions.join(', ')}</p>`;
                }
                
                if (data.features.detected_classes.length > 0) {
                    featuresHtml += `<p><strong>Classes:</strong> ${data.features.detected_classes.join(', ')}</p>`;
                }
                
                if (data.features.detected_races.length > 0) {
                    featuresHtml += `<p><strong>Races:</strong> ${data.features.detected_races.join(', ')}</p>`;
                }
                
                if (data.features.likely_scope !== 'UNKNOWN') {
                    featuresHtml += `<p><strong>Likely scope:</strong> ${data.features.likely_scope}</p>`;
                }
                
                featuresDiv.innerHTML = featuresHtml;
                container.appendChild(featuresDiv);
            }
            
            // Display rules
            if (data.rules && data.rules.length > 0) {
                const rulesContainer = document.createElement('div');
                rulesContainer.innerHTML = '<h3>Relevant Rules</h3>';
                
                data.rules.forEach(rule => {
                    const ruleCard = document.createElement('div');
                    ruleCard.className = 'rule-card';
                    
                    let ruleTypeTag = `<span class="tag">${rule.type}</span>`;
                    let ruleScopeTag = rule.scope ? `<span class="tag">${rule.scope}</span>` : '';
                    let ruleComplexityTag = rule.complexity ? `<span class="tag">${rule.complexity}</span>` : '';
                    
                    ruleCard.innerHTML = `
                        <h4 class="rule-title">${rule.title}</h4>
                        <div class="rule-meta">
                            ${ruleTypeTag}
                            ${ruleScopeTag}
                            ${ruleComplexityTag}
                            ${rule.relevance_score ? `<span class="tag">Score: ${rule.relevance_score.toFixed(1)}</span>` : ''}
                        </div>
                        <div class="rule-content">
                            ${rule.text}
                        </div>
                    `;
                    
                    rulesContainer.appendChild(ruleCard);
                });
                
                container.appendChild(rulesContainer);
            } else {
                container.innerHTML += '<p>No relevant rules found.</p>';
            }
        }
        
        function loadStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    const statsContainer = document.getElementById('stats-content');
                    
                    let statsHtml = `
                        <p><strong>Total Rules:</strong> ${data.total_rules}</p>
                        <p><strong>Total Terms:</strong> ${data.total_terms}</p>
                        <p><strong>Total Relationships:</strong> ${data.total_relationships}</p>
                    `;
                    
                    // Rule Types
                    statsHtml += '<p><strong>Rule Types:</strong></p><ul>';
                    for (const [type, count] of Object.entries(data.rule_types)) {
                        statsHtml += `<li>${type}: ${count}</li>`;
                    }
                    statsHtml += '</ul>';
                    
                    // Rule Scopes
                    statsHtml += '<p><strong>Rule Scopes:</strong></p><ul>';
                    for (const [scope, count] of Object.entries(data.rule_scopes)) {
                        statsHtml += `<li>${scope}: ${count}</li>`;
                    }
                    statsHtml += '</ul>';
                    
                    statsContainer.innerHTML = statsHtml;
                })
                .catch(error => {
                    console.error('Error loading stats:', error);
                    document.getElementById('stats-content').innerHTML = 'Error loading stats';
                });
        }
    </script>
</body>
</html>
        ''')
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
