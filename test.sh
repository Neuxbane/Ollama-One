#!/bin/bash

# Google AI Proxy Endpoint Test Script
BASE_URL="http://127.0.0.1:11434"
PASS=0
FAIL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Google AI Proxy Endpoint Test Suite                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Checking if server is running...${NC}"
if ! curl -s "$BASE_URL/" > /dev/null 2>&1; then
    echo -e "${RED}✗ Server is not running at $BASE_URL${NC}"
    echo -e "${YELLOW}Start the server with: ./build.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"
echo ""

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

# Test 1: Root status endpoint
test_endpoint "GET" "/" "" "200" "Root endpoint (status check)"

# Test 2: Models list endpoint
test_endpoint "GET" "/v1/models" "" "200" "List models (/v1/models)"

# Test 3: Single model endpoint
test_endpoint "GET" "/v1/models/google/gemini-2.0-flash" "" "200" "Single model info"

# Test 4: Chat completions non-streaming
test_endpoint "POST" "/v1/chat/completions" \
    '{"model":"google/gemini-2.0-flash","messages":[{"role":"user","content":"Say hi in one word"}],"stream":false}' \
    "200" \
    "Chat completions (non-streaming)"

# Test 5: Chat completions streaming
echo -e "${BLUE}Testing: Chat completions (streaming SSE)${NC}"
echo -e "  Endpoint: POST /v1/chat/completions"
response=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"google/gemini-2.0-flash","messages":[{"role":"user","content":"Count to 3"}],"stream":true}' \
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

# Test 6: CORS OPTIONS request
test_endpoint "OPTIONS" "/v1/chat/completions" "" "204" "CORS preflight request"

# Print summary
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "Tests completed: $((PASS + FAIL))"
echo -e "${GREEN}Passed: $PASS${NC}"
if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Failed: $FAIL${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
