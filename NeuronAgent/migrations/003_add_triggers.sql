-- Triggers for updated_at
CREATE OR REPLACE FUNCTION neurondb_agent.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER agents_updated_at BEFORE UPDATE ON neurondb_agent.agents
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_updated_at();

CREATE TRIGGER tools_updated_at BEFORE UPDATE ON neurondb_agent.tools
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_updated_at();

CREATE TRIGGER jobs_updated_at BEFORE UPDATE ON neurondb_agent.jobs
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_updated_at();

-- Trigger for session last_activity_at
CREATE OR REPLACE FUNCTION neurondb_agent.update_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE neurondb_agent.sessions
    SET last_activity_at = NOW()
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER messages_session_activity AFTER INSERT ON neurondb_agent.messages
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_session_activity();

