
all: up

up:
	@if [ -z "`docker ps -q -f name=rag_container`" ]; then \
        echo "\033[0;32mInitiating startup process for 1337 RAG App ...\033[0m"; \
        docker build -t rag_app:1.0.0 . | true && echo "\033[0;34m1337 RAG App image built successfully.\033[0m"; \
        docker run -it --name rag_container rag_app:1.0.0 | true && echo "\033[0;34m1337 RAG App container started successfully.\033[0m"; \
        echo "\033[0;32mStartup process for 1337 RAG App completed.\033[0m"; \
		docker exec -it rag_container bash; \
    else \
        echo "\033[0;33m1337 RAG App is already running. No action required.\033[0m"; \
    fi

down:
	@if [ -n "`docker ps -q -f name=rag_container`" ]; then \
        echo "\033[0;31mInitiating shutdown process for 1337 RAG App ...\033[0m"; \
        docker stop rag_container | true && echo "\033[0;34m1337 RAG App container stopped successfully.\033[0m"; \
        docker rm rag_container | true && echo "\033[0;34m1337 RAG App container removed successfully.\033[0m"; \
        docker rmi rag_app:1.0.0 | true && echo "\033[0;34m1337 RAG App image removed successfully.\033[0m"; \
        echo "\033[0;31mShutdown process for 1337 RAG App completed.\033[0m"; \
    else \
        echo "\033[0;33m1337 RAG App is not running. No action required.\033[0m"; \
    fi

clean: down
	docker system prune -af

re: down up