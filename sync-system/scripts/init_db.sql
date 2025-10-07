CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    rate_limit_per_minute INTEGER DEFAULT 60,
    total_requests INTEGER DEFAULT 0,
    last_used_at TIMESTAMP
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_active ON api_keys(is_active);

CREATE TABLE IF NOT EXISTS collaboration_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_key_id UUID REFERENCES api_keys(id),
    query TEXT NOT NULL,
    context TEXT,
    num_agents INTEGER NOT NULL,
    max_rounds INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    total_rounds INTEGER NOT NULL,
    total_messages INTEGER NOT NULL,
    computation_time FLOAT NOT NULL,
    confidence FLOAT,
    consensus_level FLOAT,
    final_answer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_created_at ON collaboration_sessions(created_at);
CREATE INDEX idx_sessions_api_key ON collaboration_sessions(api_key_id);
CREATE INDEX idx_sessions_success ON collaboration_sessions(success);

CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES collaboration_sessions(id),
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    round_number INTEGER,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_session ON metrics(session_id);
CREATE INDEX idx_metrics_type ON metrics(metric_type);
CREATE INDEX idx_metrics_recorded_at ON metrics(recorded_at);

CREATE TABLE IF NOT EXISTS agent_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES collaboration_sessions(id),
    agent_id INTEGER NOT NULL,
    role VARCHAR(100),
    messages_sent INTEGER DEFAULT 0,
    contribution_score FLOAT,
    avg_gap_magnitude FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_agent_perf_session ON agent_performance(session_id);
CREATE INDEX idx_agent_perf_agent ON agent_performance(agent_id);

CREATE TABLE IF NOT EXISTS training_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    checkpoint_name VARCHAR(255) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL,
    epoch INTEGER,
    step INTEGER,
    loss FLOAT,
    metrics JSONB,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_checkpoints_name ON training_checkpoints(checkpoint_name);
CREATE INDEX idx_checkpoints_type ON training_checkpoints(model_type);
CREATE INDEX idx_checkpoints_created_at ON training_checkpoints(created_at);

CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status VARCHAR(20) NOT NULL,
    avg_response_time FLOAT,
    total_sessions_24h INTEGER,
    success_rate_24h FLOAT,
    active_connections INTEGER,
    memory_usage_mb FLOAT,
    cpu_usage_percent FLOAT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_health_recorded_at ON system_health(recorded_at);

INSERT INTO api_keys (key_hash, name, rate_limit_per_minute)
VALUES (
    '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92',
    'Admin Key',
    1000
) ON CONFLICT (key_hash) DO NOTHING;


CREATE OR REPLACE VIEW daily_success_rate AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_sessions,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_sessions,
    ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate,
    ROUND(AVG(computation_time), 2) as avg_computation_time,
    ROUND(AVG(total_rounds), 2) as avg_rounds
FROM collaboration_sessions
GROUP BY DATE(created_at)
ORDER BY date DESC;

CREATE OR REPLACE VIEW api_key_usage AS
SELECT 
    ak.name,
    ak.total_requests,
    ak.rate_limit_per_minute,
    COUNT(cs.id) as sessions_count,
    MAX(cs.created_at) as last_session_at,
    ROUND(AVG(cs.computation_time), 2) as avg_session_time
FROM api_keys ak
LEFT JOIN collaboration_sessions cs ON ak.id = cs.api_key_id
GROUP BY ak.id, ak.name, ak.total_requests, ak.rate_limit_per_minute;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO syncuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO syncuser;

SELECT 'Database initialized successfully!' as status;
