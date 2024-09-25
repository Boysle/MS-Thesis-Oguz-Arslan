# Welcome to Oguz Arslan's Master Thesis Repository

### Current Description of the Project

The aim of this project is to develop a user interface framework tailored to esports titles and traditional sports, specifically those played between two teams and involving elements of time and location with clear winning conditions. The core functionality of this framework will be based on an ML model designed to predict scoring events or other significant moments that would give one team an advantage over the other.

The end users of this system include coaches, players, and fans who are interested in exploring "what if" scenarios within a game. By interacting with the system, users can manipulate specific aspects of a game's snapshot, such as changing a player's position, to observe how these changes affect the winning probability for a team. This interactive tool provides valuable insights, allowing users to test different strategies and understand the impact of various in-game decisions.

As a demonstration and proof of concept, we will train a GNN goal prediction model for the popular esports title Rocket League, focusing on predicting the likelihood of a goal occurring within 10 seconds of a particular in-game snapshot. The user interface will be modeled around this specific use case.

Ultimately, the tool will be designed to be adaptable, allowing users to hook their own predictive models into the defined structure. The framework will support the addition of various features relevant to specific games, such as player health in games where this is a factor, and the ability to load in maps with linked locations. After extensive research into both esports and traditional sports, it was determined that this tool is most applicable to esports genres like MOBAs and FPS games, as well as traditional sports played on a field or court with a ball.**
