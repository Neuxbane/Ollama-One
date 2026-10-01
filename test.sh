#!/bin/bash

# Ollama-One Endpoint Test Script
# Tests all available endpoints in the Ollama-One proxy

BASE_URL="http://127.0.0.1:11434"
PASS=0
FAIL=0

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Ollama-One Endpoint Test Suite                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}Checking if server is running...${NC}"
if ! curl -s "$BASE_URL/" > /dev/null 2>&1; then
    echo -e "${RED}✗ Server is not running at $BASE_URL${NC}"
    echo -e "${YELLOW}Start the server with: ./build.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"
echo ""

# Helper function to test endpoints
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local expected_code=$4
    local description=$5
    
    echo -e "${BLUE}Testing: $description${NC}"
    echo -e "  Endpoint: $method $endpoint"
    
    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$BASE_URL$endpoint")
    elif [ "$method" = "OPTIONS" ]; then
        response=$(curl -s -w "\n%{http_code}" -X OPTIONS \
            -H "Content-Type: application/json" \
            "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint")
    fi
    
    # Extract status code and body
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    echo "  Status Code: $http_code"
    
    if [ "$http_code" = "$expected_code" ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS++))
    else
        echo -e "${RED}✗ FAIL (expected $expected_code)${NC}"
        ((FAIL++))
    fi
    
    if [ -n "$body" ] && [ ${#body} -lt 500 ]; then
        echo "  Response: $body"
    elif [ -n "$body" ]; then
        echo "  Response: ${body:0:200}... (truncated)"
    fi
    echo ""
}

# Test 1: Root endpoint
test_endpoint "GET" "/" "" "200" "Root endpoint (server status)"

# Test 2: Version endpoint
test_endpoint "GET" "/api/version" "" "200" "Version endpoint"

# Test 3: Tags endpoint (list models)
test_endpoint "GET" "/api/tags" "" "200" "Tags endpoint (list models)"

# Test 4: Show endpoint with valid model
test_endpoint "POST" "/api/show" \
    '{"name":"gemini-2.0-flash"}' \
    "200" \
    "Show endpoint (model info)"

# Test 5: Show endpoint with alternative field
test_endpoint "POST" "/api/show" \
    '{"model":"gemini-2.0-flash"}' \
    "200" \
    "Show endpoint with model field"

# Test 6: Chat endpoint (non-streaming)
test_endpoint "POST" "/api/chat" \
    '{"model":"gemini-2.0-flash","messages":[{"role":"user","content":"What is 2+2?"}],"stream":false}' \
    "200" \
    "Chat endpoint (non-streaming)"

# Test 7: Chat endpoint (streaming)
echo -e "${BLUE}Testing: Chat endpoint (streaming)${NC}"
echo -e "  Endpoint: POST /api/chat"
response=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"gemini-2.0-flash","messages":[{"role":"user","content":"Hello"}],"stream":true}' \
    "$BASE_URL/api/chat")
http_code=$(echo "$response" | tail -n1)
echo "  Status Code: $http_code"
if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL (expected 200)${NC}"
    ((FAIL++))
fi
echo ""

# Test 8: Generate endpoint (non-streaming)
test_endpoint "POST" "/api/generate" \
    '{"model":"gemini-2.0-flash","prompt":"What is AI?","stream":false}' \
    "200" \
    "Generate endpoint (non-streaming)"

# Test 9: Generate endpoint (streaming)
echo -e "${BLUE}Testing: Generate endpoint (streaming)${NC}"
echo -e "  Endpoint: POST /api/generate"
response=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"gemini-2.0-flash","prompt":"Hello world","stream":true}' \
    "$BASE_URL/api/generate")
http_code=$(echo "$response" | tail -n1)
echo "  Status Code: $http_code"
if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL (expected 200)${NC}"
    ((FAIL++))
fi
echo ""

# Test 10: OpenAI compatible endpoint (non-streaming)
test_endpoint "POST" "/v1/chat/completions" \
    '{"model":"gpt-4o","messages":[{"role":"user","content":"What is machine learning?"}],"stream":false}' \
    "200" \
    "OpenAI compatible endpoint (non-streaming)"

# Test 11: OpenAI compatible endpoint (streaming)
echo -e "${BLUE}Testing: OpenAI compatible endpoint (streaming)${NC}"
echo -e "  Endpoint: POST /v1/chat/completions"
response=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}],"stream":true}' \
    "$BASE_URL/v1/chat/completions")
http_code=$(echo "$response" | tail -n1)
echo "  Status Code: $http_code"
if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL (expected 200)${NC}"
    ((FAIL++))
fi
echo ""

# Test 12: Chat with tools
test_endpoint "POST" "/api/chat" \
    '{"model":"gemini-2.0-flash","messages":[{"role":"user","content":"What is the weather?"}],"stream":false,"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}]}' \
    "200" \
    "Chat endpoint with tools"

# Test 13: CORS preflight request
test_endpoint "OPTIONS" "/api/chat" "" "204" "CORS preflight (OPTIONS request)"

# Test 14: Invalid JSON (should fail)
echo -e "${BLUE}Testing: Invalid JSON request${NC}"
echo -e "  Endpoint: POST /api/chat"
response=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{invalid json}' \
    "$BASE_URL/api/chat")
http_code=$(echo "$response" | tail -n1)
echo "  Status Code: $http_code"
if [ "$http_code" = "400" ]; then
    echo -e "${GREEN}✓ PASS (correctly rejected invalid JSON)${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL (expected 400)${NC}"
    ((FAIL++))
fi
echo ""

# Test 15: Favicon request (should 404)
echo -e "${BLUE}Testing: Favicon endpoint (404 expected)${NC}"
echo -e "  Endpoint: GET /favicon.ico"
response=$(curl -s -w "\n%{http_code}" "$BASE_URL/favicon.ico")
http_code=$(echo "$response" | tail -n1)
echo "  Status Code: $http_code"
if [ "$http_code" = "404" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL (expected 404)${NC}"
    ((FAIL++))
fi
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     Test Summary                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
total=$((PASS + FAIL))
echo -e "Total Tests: $total"
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
