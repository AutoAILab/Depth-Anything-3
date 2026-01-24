# API Development Guidelines

## FastAPI Backend
- Use FastAPI for REST API development
- Implement proper request/response models with Pydantic
- Use async/await for I/O operations
- Implement dependency injection for shared resources

## Endpoint Design
- Use RESTful URL patterns
- Implement proper HTTP status codes
- Use JSON for data exchange
- Document APIs with OpenAPI/Swagger

## Request Handling
- Validate input data with Pydantic models
- Implement proper error responses
- Handle file uploads with `UploadFile`
- Support streaming responses for large data

## Authentication and Security
- Implement API key authentication
- Use HTTPS in production
- Validate and sanitize all inputs
- Implement rate limiting and CORS

## Performance
- Use background tasks for long operations
- Implement caching with Redis or similar
- Optimize database queries
- Use connection pooling

## Monitoring and Logging
- Implement health check endpoints
- Log requests and responses
- Use metrics collection (Prometheus, etc.)
- Implement proper error tracking

## Testing
- Use `pytest` with `httpx` for API testing
- Test all endpoints and error cases
- Implement integration tests
- Use test clients for isolated testing

## Deployment
- Use Docker for containerization
- Implement proper environment configuration
- Use reverse proxies (nginx, etc.)
- Implement graceful shutdown handling