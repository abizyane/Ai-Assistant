C_RED = \033[1;31m
C_GREEN = \033[1;32m
C_YELLOW = \033[1;33m
C_RESET = \033[0m

IMAGE_NAME = rag-app
CONTAINER_NAME = rag-container
KNOWLEDGE_BASE_PATH = ./rag/knowledge_base

all: build up

build:
	@echo "${C_GREEN}Building RAG application...${C_RESET}"
	@docker-compose build
	@echo "${C_GREEN}Build completed!${C_RESET}"

up:
	@echo "${C_GREEN}Starting RAG application...${C_RESET}"
	@docker-compose up
	@echo "${C_GREEN}Application exited.${C_RESET}"

down:
	@echo "${C_RED}Stopping RAG application...${C_RESET}"
	@docker-compose down
	@echo "${C_RED}Stopped!${C_RESET}"

state:
	@echo "${C_YELLOW}Checking container state...${C_RESET}"
	@docker-compose ps

logs:
	@echo "${C_YELLOW}Showing container logs...${C_RESET}"
	@docker-compose logs -f

clean: down
	@echo "${C_RED}Cleaning...${C_RESET}"
	@docker-compose down --rmi all -v
	@echo "${C_RED}Cleaning Done!${C_RESET}"

fclean: clean
	@echo "${C_RED}Full cleaning...${C_RESET}"
	@docker system prune -f
	@echo "${C_RED}Full cleaning Done!${C_RESET}"

rebuild: clean build up

help:
	@echo "Usage: make [command]"
	@echo "Commands:"
	@echo "  build: Build the RAG application Docker image"
	@echo "  up: Run the RAG application in a container"
	@echo "  down: Stop the RAG application container"
	@echo "  state: Check the state of the container"
	@echo "  logs: Show the logs of the container"
	@echo "  clean: Remove the container and image"
	@echo "  fclean: Clean and remove Docker cache"
	@echo "  rebuild: Clean, rebuild and run the application"
	@echo "  help: Show this help message"

.PHONY: all build up down state logs clean fclean rebuild help
