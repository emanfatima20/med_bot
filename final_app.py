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
tokenizer.pad_token = tokenizer.eos_token  # Important for some models

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# ============================================
# 🔧 Create streamer for real-time output
# ============================================
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=10.0)

# ============================================
# 🔧 Optimized pipeline with streaming support
# ============================================
print("Creating pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    max_new_tokens=80,
    temperature=0.3,
    top_p=0.85,
    do_sample=True,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipe)

# ============================================
# 🔍 Search tool
# ============================================
search = DuckDuckGoSearchRun()
search_cache = {}

# ============================================
# 🎯 Improved prompt engineering
# ============================================
def create_smart_prompt(query, context=""):
    """Create optimized prompts that prevent repetition"""
    
    if context:
        return f"""Based on the research context below, provide a clear summary:

Research Context: {context}

Question: {query}

Answer concisely and directly:"""
    
    else:
        return f"""Provide a clear, direct answer to this question:

Question: {query}

Answer:"""

# ============================================
# 🔄 Streaming generation function
# ============================================
def generate_with_streaming(prompt, use_streaming=True):
    """
    Generate text with optional streaming
    """
    if not use_streaming:
        # Fallback to non-streaming
        response = llm.invoke(prompt)
        return response if isinstance(response, str) else response[0]['generated_text']
    
    # Streaming generation
    generation_kwargs = {
        "max_new_tokens": 80,
        "temperature": 0.3,
        "top_p": 0.85,
        "do_sample": True,
        "repetition_penalty": 1.2,
        "streamer": streamer,
    }
    
    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs={
        **generation_kwargs,
        **tokenizer(prompt, return_tensors="pt")
    })
    thread.daemon = True
    thread.start()
    
    # Collect streamed text
    generated_text = ""
    print("💡 Answer: ", end="", flush=True)
    
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_text += new_text
    
    thread.join()
    return generated_text

# ============================================
# ⚡ Optimized medical agent with streaming
# ============================================
def medical_agent_optimized(query, use_streaming=True):
    """
    Optimized version with streaming support
    """
    start_time = time.time()
    
    # Complexity check
    complex_keywords = {"latest", "research", "study", "clinical", "recent", "new", "findings"}
    is_complex = any(keyword in query.lower() for keyword in complex_keywords)

    context_summary = ""
    if is_complex:
        cache_key = query.lower()[:50]
        if cache_key in search_cache:
            context_summary = search_cache[cache_key]
            print("📚 Using cached search")
        else:
            try:
                print("🔍 Searching...", end=" ", flush=True)
                search_start = time.time()
                search_results = search.run(query)
                context_summary = search_results[:350] if len(search_results) > 350 else search_results
                search_cache[cache_key] = context_summary
                print(f"✅ ({time.time() - search_start:.1f}s)")
            except Exception as e:
                print(f"❌ Search failed")
                context_summary = ""

    # Use optimized prompt
    prompt = create_smart_prompt(query, context_summary if is_complex else "")
    
    print(f"\n{'='*60}")
    print(f"❓ Question: {query}")
    
    gen_start = time.time()
    try:
        if use_streaming:
            response_text = generate_with_streaming(prompt, use_streaming=True)
            print()  # New line after streaming
        else:
            print("💡 Generating...", end=" ", flush=True)
            response_text = generate_with_streaming(prompt, use_streaming=False)
            print(f"✅ ({time.time() - gen_start:.1f}s)")
            print(f"💡 Answer: {response_text}")
        
        # Advanced cleaning
        cleaned_text = clean_response(response_text, prompt, query)
        
        if not use_streaming:
            print(f"💡 Answer: {cleaned_text}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        cleaned_text = "I apologize, but I encountered an error generating a response."

    total_time = time.time() - start_time
    print(f"\n⏱️  Total time: {total_time:.1f}s")
    print("="*60)
    
    return cleaned_text

def clean_response(response_text, prompt, original_query):
    """Advanced response cleaning to remove repetitions and artifacts"""
    
    # Remove the prompt from response
    if prompt in response_text:
        response_text = response_text.replace(prompt, "")
    
    # Remove repeated questions
    if original_query.lower() in response_text.lower():
        # Find where the answer actually starts
        query_pos = response_text.lower().find(original_query.lower())
        if query_pos > 0:
            response_text = response_text[query_pos + len(original_query):]
    
    # Remove common artifacts and repetitions
    cleaning_patterns = [
        "Question:",
        "Answer:",
        "Context:",
        "Based on the research context",
        "Provide a clear summary",
        "Answer concisely and directly",
        "The present study aimed to",
        "The objective was to",
        "The study population was",
        "The participants were",
        "The results showed that",
        "The study was approved by",
    ]
    
    for pattern in cleaning_patterns:
        response_text = response_text.replace(pattern, "")
    
    # Split into sentences and remove duplicates
    sentences = [s.strip() for s in response_text.split('.') if s.strip()]
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        # Simple deduplication based on first few words
        key = ' '.join(sentence.split()[:6]).lower()
        if key not in seen_sentences and len(sentence) > 10:
            unique_sentences.append(sentence)
            seen_sentences.add(key)
    
    cleaned_text = '. '.join(unique_sentences[:4])  # Limit to 4 sentences
    cleaned_text = cleaned_text.strip()
    
    # Final cleanup
    cleaned_text = ' '.join(cleaned_text.split())
    
    if not cleaned_text or len(cleaned_text) < 10:
        return "I couldn't generate a proper response. Please try rephrasing your question."
    
    return cleaned_text

# ============================================
# 🔄 Simple streaming alternative (if above doesn't work)
# ============================================
def simple_streaming_generation(query, use_streaming=True):
    """
    Alternative simpler streaming approach
    """
    prompt = create_smart_prompt(query)
    
    if not use_streaming:
        response = llm.invoke(prompt)
        response_text = response if isinstance(response, str) else response[0]['generated_text']
        return clean_response(response_text, prompt, query)
    
    # Simulated streaming (character by character)
    print("💡 Answer: ", end="", flush=True)
    response = llm.invoke(prompt)
    response_text = response if isinstance(response, str) else response[0]['generated_text']
    cleaned_text = clean_response(response_text, prompt, query)
    
    # Print with streaming effect
    for char in cleaned_text:
        print(char, end="", flush=True)
        time.sleep(0.02)  # Adjust speed here
    print()
    
    return cleaned_text

# ============================================
# 🚀 Main execution with streaming options
# ============================================
if __name__ == "__main__":
    print("🚀 Starting OPTIMIZED medical assistant with STREAMING...")
    print("💬 Choose mode:")
    print("1. Real streaming (if supported)")
    print("2. Simulated streaming (always works)")
    
    mode = 2  # Change to 1 for real streaming, 2 for simulated
    
    # Test questions
    test_questions = [
        "What is the function of insulin in the human body?",
        "What are the latest research findings on insulin pumps?",
        "Why is exercise important for mental health?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🧪 Test {i}/{len(test_questions)}")
        
        if mode == 1:
            # Try real streaming
            try:
                medical_agent_optimized(question, use_streaming=True)
            except Exception as e:
                print(f"Real streaming failed, falling back to simulated: {e}")
                medical_agent_optimized(question, use_streaming=False)
        else:
            # Use simulated streaming
            print(f"\n{'='*60}")
            print(f"❓ Question: {question}")
            start_time = time.time()
            
            # Complexity check for search
            complex_keywords = {"latest", "research", "study", "clinical", "recent", "new", "findings"}
            is_complex = any(keyword in question.lower() for keyword in complex_keywords)
            
            if is_complex:
                try:
                    print("🔍 Searching...", end=" ", flush=True)
                    search_start = time.time()
                    search_results = search.run(question)
                    context_summary = search_results[:350] if len(search_results) > 350 else search_results
                    print(f"✅ ({time.time() - search_start:.1f}s)")
                except:
                    print("❌ Search failed")
                    context_summary = ""
            
            # Use simulated streaming
            response = simple_streaming_generation(question, use_streaming=True)
            
            total_time = time.time() - start_time
            print(f"⏱️  Total time: {total_time:.1f}s")
            print("="*60)