from rag import setup_rag
from chain import get_ux_advice
from dotenv import find_dotenv, load_dotenv
 
# _ = load_dotenv(find_dotenv())

def main():
    setup_rag()
    while True:
        question = input("\nQuestion UX: ")
        
        if question.lower() == 'quit':
            break
            
        advice = get_ux_advice(question)
        
        print(f"\n{advice}")

if __name__ == "__main__":
    main()