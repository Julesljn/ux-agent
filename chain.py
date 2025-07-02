from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import config
from rag import search_rules
from prompt import ux_prompt, query_rewrite_prompt

llm = ChatOllama(
    model=config.OLLAMA_MODEL,
    base_url=config.OLLAMA_URL,
    temperature=config.TEMPERATURE
)

rewrite_chain = query_rewrite_prompt | llm | StrOutputParser()

def process_question_and_get_context(inputs):
    original_question = inputs["question"]
    
    rewritten_keywords = rewrite_chain.invoke({"question": original_question})
    
    rules = search_rules(rewritten_keywords)
    
    print(f"{original_question}")
    print(f"Mots-clés : {rewritten_keywords.strip()}")
    
    print(f"Règles trouvées ({len(rules)}):")
    for rule in rules:
        content = rule['content']
        titre = content.split('titre :')[1].split(' / catégorie')[0].strip()
        print(f"  - {titre}")

    context_rules = []
    for rule in rules:
        content = rule['content']
        titre = content.split('titre :')[1].split(' / catégorie')[0].strip()
        description = content.split(' / ')[-1].strip()
        context_rules.append(f"- {titre}: {description}")
    
    return "\n".join(context_rules)

ux_chain = (
    RunnablePassthrough.assign(context=process_question_and_get_context)
    | ux_prompt 
    | llm 
    | StrOutputParser()
)

def get_ux_advice(question):
    try:
        result = ux_chain.invoke({"question": question})
        return result
    except Exception as e:
        return f"Erreur: {e}"

