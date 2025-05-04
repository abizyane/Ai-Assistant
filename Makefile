C_RED = \033[1;31m
C_GREEN = \033[1;32m
C_YELLOW = \033[1;33m
C_RESET = \033[0m

include .env
export

IMAGE_NAME = rag-app
CONTAINER_NAME = rag-container
KNOWLEDGE_BASE_PATH = ./rag/knowledge_base

all: build run

build:
	@echo "${C_GREEN}Building RAG application...${C_RESET}"
	@docker build -t $(IMAGE_NAME) \
        --build-arg GEMINI_API_KEY=$(GEMINI_API_KEY) \
        .
	@echo "${C_GREEN}Build completed!${C_RESET}"

run:
	@echo "${C_GREEN}Starting RAG application...${C_RESET}"
	@docker run -it --rm \
        --name $(CONTAINER_NAME) \
        -e GEMINI_API_KEY=$(GEMINI_API_KEY) \
        -v $(KNOWLEDGE_BASE_PATH):/app/rag/knowledge_base \
        $(IMAGE_NAME)
	@echo "${C_GREEN}Application exited.${C_RESET}"

stop:
	@echo "${C_RED}Stopping RAG application...${C_RESET}"
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@echo "${C_RED}Stopped!${C_RESET}"

state:
	@echo "${C_YELLOW}Checking container state...${C_RESET}"
	@docker ps -a | grep $(CONTAINER_NAME) || echo "Container not running"

logs:
	@echo "${C_YELLOW}Showing container logs...${C_RESET}"
	@docker logs -f $(CONTAINER_NAME)

clean: stop
	@echo "${C_RED}Cleaning...${C_RESET}"
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "${C_RED}Cleaning Done!${C_RESET}"

fclean: clean
	@echo "${C_RED}Full cleaning...${C_RESET}"
	@docker system prune -f
	@echo "${C_RED}Full cleaning Done!${C_RESET}"

rebuild: clean build run

help:
	@echo "Usage: make [command]"
	@echo "Commands:"
	@echo "  build: Build the RAG application Docker image"
	@echo "  run: Run the RAG application in a container"
	@echo "  stop: Stop the RAG application container"
	@echo "  state: Check the state of the container"
	@echo "  logs: Show the logs of the container"
	@echo "  clean: Remove the container and image"
	@echo "  fclean: Clean and remove Docker cache"
	@echo "  rebuild: Clean, rebuild and run the application"
	@echo "  help: Show this help message"

.PHONY: all build run stop state logs clean fclean rebuild help