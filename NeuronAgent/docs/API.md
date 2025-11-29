# NeuronAgent API Documentation

## Base URL

```
http://localhost:8080/api/v1
```

## Authentication

All API requests require authentication using an API key in the Authorization header:

```
Authorization: Bearer <api_key>
```

## Endpoints

### Agents

#### Create Agent
```
POST /api/v1/agents
```

Request body:
```json
{
  "name": "my-agent",
  "description": "A helpful agent",
  "system_prompt": "You are a helpful assistant.",
  "model_name": "gpt-4",
  "enabled_tools": ["sql", "http"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

#### List Agents
```
GET /api/v1/agents
```

#### Get Agent
```
GET /api/v1/agents/{id}
```

#### Delete Agent
```
DELETE /api/v1/agents/{id}
```

### Sessions

#### Create Session
```
POST /api/v1/sessions
```

Request body:
```json
{
  "agent_id": "uuid",
  "external_user_id": "user123",
  "metadata": {}
}
```

#### Get Session
```
GET /api/v1/sessions/{id}
```

### Messages

#### Send Message
```
POST /api/v1/sessions/{session_id}/messages
```

Request body:
```json
{
  "role": "user",
  "content": "Hello, how are you?",
  "stream": false
}
```

#### Get Messages
```
GET /api/v1/sessions/{session_id}/messages
```

### WebSocket

#### Connect to WebSocket
```
WS /ws?session_id={session_id}
```

Send messages:
```json
{
  "content": "Hello"
}
```

Receive responses:
```json
{
  "type": "response",
  "content": "Hello! How can I help you?",
  "complete": true
}
```

