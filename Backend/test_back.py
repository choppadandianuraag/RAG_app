#!/usr/bin/env python3
"""
Test script for RAG backend
Run this to verify your setup is working correctly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_rag_engine():
    """Test RAG engine initialization and queries"""
    print("\n" + "="*80)
    print("🧪 TESTING RAG ENGINE")
    print("="*80 + "\n")
    
    try:
        from rag_engine import RAGEngine
        
        # Check environment variables
        print("1️⃣ Checking environment variables...")
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            print("   ❌ GROQ_API_KEY not found")
            print("   💡 Set it with: export GROQ_API_KEY='your_key_here'")
            return False
        print("   ✅ GROQ_API_KEY found")
        
        pdf_path = os.environ.get("PDF_PATH", "/Users/anuraag/Python/RAG App/FAQ Data.pdf")
        if not os.path.exists(pdf_path):
            print(f"   ❌ PDF not found at: {pdf_path}")
            print(f"   💡 Add your PDF to: {pdf_path}")
            return False
        print(f"   ✅ PDF found at: {pdf_path}")
        
        # Initialize engine
        print("\n2️⃣ Initializing RAG engine...")
        engine = RAGEngine()
        await engine.initialize()
        print("   ✅ RAG engine initialized successfully")
        
        # Test queries
        print("\n3️⃣ Testing queries...")
        test_queries = [
            "What is the refund processing time?",
            "What is the return policy?",
            "How do I track my order?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            result = await engine.get_response(query)
            print(f"   Answer: {result['answer'][:150]}...")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            print(f"   Retrieved docs: {len(result['docs'])}")
        
        print("\n   ✅ All queries completed successfully")
        
        return True
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_server():
    """Test if API server is running and responding"""
    print("\n" + "="*80)
    print("🌐 TESTING API SERVER")
    print("="*80 + "\n")
    
    try:
        import requests
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        print("1️⃣ Testing health endpoint...")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ Health check passed: {response.json()}")
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("   ❌ Cannot connect to server")
            print("   💡 Make sure server is running: python main.py")
            return False
        
        # Test categories endpoint
        print("\n2️⃣ Testing categories endpoint...")
        response = requests.get(f"{base_url}/categories")
        if response.status_code == 200:
            categories = response.json()
            print(f"   ✅ Got {len(categories)} categories")
            for cat in categories[:3]:
                print(f"      - {cat['icon']} {cat['name']}")
        
        # Test chat endpoint
        print("\n3️⃣ Testing chat endpoint...")
        test_message = "What is the refund processing time?"
        response = requests.post(
            f"{base_url}/chat",
            json={
                "message": test_message,
                "history": [],
                "category": "refund"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Chat response received")
            print(f"   Query: {test_message}")
            print(f"   Answer: {result['response'][:150]}...")
        else:
            print(f"   ❌ Chat failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
        
        print("\n   ✅ All API tests passed")
        return True
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test if all required packages are installed"""
    print("\n" + "="*80)
    print("📦 TESTING DEPENDENCIES")
    print("="*80 + "\n")
    
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('langchain', 'LangChain'),
        ('langchain_groq', 'LangChain Groq'),
        ('faiss', 'FAISS (CPU)'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('pypdf', 'PyPDF'),
    ]
    
    all_installed = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\n   💡 Install missing packages: pip install -r requirements.txt")
        return False
    
    print("\n   ✅ All dependencies installed")
    return True

async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🚀 RAG BACKEND TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Test 1: Dependencies
    results['dependencies'] = test_dependencies()
    
    # Test 2: RAG Engine (only if dependencies are OK)
    if results['dependencies']:
        results['rag_engine'] = await test_rag_engine()
    else:
        results['rag_engine'] = False
        print("\n⏭️  Skipping RAG engine test (dependencies missing)")
    
    # Test 3: API Server
    print("\n💡 Make sure the API server is running in another terminal:")
    print("   python main.py")
    input("\nPress Enter when server is ready (or Ctrl+C to skip)...")
    
    try:
        results['api_server'] = await test_api_server()
    except KeyboardInterrupt:
        print("\n⏭️  Skipping API server test")
        results['api_server'] = None
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80 + "\n")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else ("⏭️  SKIP" if result is None else "❌ FAIL")
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(r for r in results.values() if r is not None)
    
    if all_passed:
        print("\n🎉 All tests passed! Your backend is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Start the server: python main.py")
        print("   2. Open docs: http://localhost:8000/docs")
        print("   3. Connect your frontend")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
        sys.exit(0)