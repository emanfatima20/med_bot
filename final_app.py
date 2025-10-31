import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from langchain_huggingface import HuggingFacePipeline
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
import time
from threading import Thread

# ============================================
# ⚙️ Device setup & Optimization
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# ============================================
# 🧠 Load model with optimizations
# ============================================
model_id = "./smollm2_pubmed_full_v3"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# ============================================
# 🔧 Optimized pipeline
# ============================================
print("Creating pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    max_new_tokens=250,  # Increased to 250 tokens
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.2,  # Stronger repetition penalty
    pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipe)

# ============================================
# 🔍 Tools
# ============================================
search_tool = DuckDuckGoSearchRun()
search_cache = {}

# ============================================
# 🛠️ Agent Tool System
# ============================================
def should_use_search(query):
    """Determine if search tool should be used"""
    search_keywords = {
        "latest", "recent", "new", "current", "today", "now", "update", 
        "research", "study", "findings", "discovery", "breakthrough",
        "news", "trending", "2024", "2025", "statistics", "data"
    }
    return any(keyword in query.lower() for keyword in search_keywords)

def use_search_tool(query):
    """Use search tool and return context"""
    print("🛠️ Using: DuckDuckGo Search")
    try:
        search_start = time.time()
        results = search_tool.run(query)
        print(f"✅ Search completed in {time.time() - search_start:.1f}s")
        return results[:1000] if len(results) > 1000 else results
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return ""

# ============================================
# 🎯 Optimized Prompts for SmolLM2
# ============================================
def create_agent_prompt(query, search_context=""):
    """Create prompts that work well with SmolLM2"""
    
    if search_context:
        return f"""<|system|>
You are a helpful AI assistant. Use the provided search context to answer the question thoroughly and accurately.

SEARCH CONTEXT:
{search_context}

QUESTION:
{query}

INSTRUCTIONS:
- Provide a comprehensive, detailed answer based on the search context
- Structure your response with clear paragraphs
- Include key facts, findings, and relevant information
- Avoid repetition and stay focused on the question
- Write in clear, natural English
</|system|>
<|user|>
{query}
</|user|>
<|assistant|>"""
    
    else:
        return f"""<|system|>
You are a helpful AI assistant. Answer the question thoroughly and accurately.

QUESTION:
{query}

INSTRUCTIONS:
- Provide a comprehensive, detailed answer
- Structure your response with clear paragraphs  
- Include key facts and relevant information
- Avoid repetition and stay focused on the question
- Write in clear, natural English
- Aim for about 200-250 words
</|system|>
<|user|>
{query}
</|user|>
<|assistant|>"""

# ============================================
# 🔄 Streaming Generation
# ============================================
def generate_with_streaming(prompt):
    """Generate text with streaming output"""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=30.0)
    
    generation_kwargs = {
        "max_new_tokens": 250,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.2,
        "streamer": streamer,
    }
    
    # Start generation in separate thread
    inputs = tokenizer(prompt, return_tensors="pt")
    thread = Thread(target=model.generate, kwargs={
        **generation_kwargs,
        **inputs
    })
    thread.daemon = True
    thread.start()
    
    # Stream output
    generated_text = ""
    print("💡 Answer: ", end="", flush=True)
    
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_text += new_text
    
    thread.join()
    print()  # New line after streaming
    return generated_text

# ============================================
# 🤖 Intelligent Agent
# ============================================
def intelligent_agent(query, use_streaming=True):
    """
    Main agent that decides when to use tools and generates responses
    """
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"🤖 AGENT PROCESSING: {query}")
    print('='*70)
    
    # Step 1: Tool Selection
    search_context = ""
    if should_use_search(query):
        cache_key = query.lower()[:100]
        if cache_key in search_cache:
            search_context = search_cache[cache_key]
            print("📚 Using cached search results")
        else:
            search_context = use_search_tool(query)
            if search_context:
                search_cache[cache_key] = search_context
    
    # Step 2: Create appropriate prompt
    prompt = create_agent_prompt(query, search_context)
    
    # Step 3: Generate response
    print("🚀 Generating response...")
    gen_start = time.time()
    
    try:
        if use_streaming:
            response_text = generate_with_streaming(prompt)
        else:
            response = llm.invoke(prompt)
            response_text = response if isinstance(response, str) else response[0]['generated_text']
            print(f"💡 Answer: {response_text}")
        
        # Step 4: Clean response
        cleaned_text = clean_agent_response(response_text)
        
        total_time = time.time() - start_time
        print(f"\n✅ Agent completed in {total_time:.1f}s")
        print('='*70)
        
        return cleaned_text
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"

def clean_agent_response(response_text):
    """Clean agent response while preserving content"""
    # Remove system prompts and artifacts
    artifacts = [
        "<|system|>", "</|system|>", 
        "<|user|>", "</|user|>", 
        "<|assistant|>", "</|assistant|>",
        "INSTRUCTIONS:", "QUESTION:", "SEARCH CONTEXT:",
        "Provide a comprehensive, detailed answer",
        "Avoid repetition and stay focused",
        "Write in clear, natural English"
    ]
    
    for artifact in artifacts:
        response_text = response_text.replace(artifact, "")
    
    # Basic cleanup
    response_text = ' '.join(response_text.split())
    response_text = response_text.strip()
    
    return response_text

# ============================================
# 🔁 Agent Loop with Continuous Interaction
# ============================================
def run_agent_loop():
    """Run continuous agent interaction loop"""
    print("🚀 Starting Intelligent Agent System...")
    print("🔧 Available Tools: DuckDuckGo Search")
    print("💬 Type 'quit' or 'exit' to end the session")
    print("="*70)
    
    while True:
        try:
            user_input = input("\n🎯 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for using the agent system. Goodbye!")
                break
                
            if not user_input:
                continue
                
            # Process with agent
            start_time = time.time()
            response = intelligent_agent(user_input, use_streaming=True)
            
            # Show token count
            token_count = len(tokenizer.encode(response))
            print(f"📊 Response tokens: {token_count}")
            
        except KeyboardInterrupt:
            print("\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ System error: {e}")
            continue

# ============================================
# 🚀 Main Execution
# ============================================
if __name__ == "__main__":
    print("🤖 Intelligent Agent System Initialized!")
    
    # Start chat loop directly
    run_agent_loop()
