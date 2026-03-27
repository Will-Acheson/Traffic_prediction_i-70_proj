# Optimizing Drive Time Algorithmically

Often when going to the mountains to ski on a weekend, it raises the question, when is the best time to leave? We can always leave at 4am to get there without worrying about traffic, but I want to be able to sleep in extra. So, when would the optimal time to leave be?
	Looking at taking time series data that is received from mediums, past and future forecasts; be able to provide an accurate timeline for the traffic patterns for the route to that location on a given day. Be able to take these data points and calculate the optimal time to leave.
	Look to use statistics and topology to create the predictions for each given point, and the optimal time to leave to go in between each point. These would be produced by both past weather and traffic data but also using the forecast to predict how the weather could affect travel.
  
-	Using statistical methods like regression trees, GLM’s, or basic regression tools at each point. Creates quick prediction models.
-	We can take these predicted models and use topology, calculate the travel with the least bumps, or traffic in this case.

	The goal would be to optimize times for travel from A to B, while looking at each intersection, or possible station to analyze what the traffic is on that given day based of previous data, and know conditions for the future.
-	Publicly available traffic data from Colorado department of transportation
-	Publicly available weather data from National weather service
