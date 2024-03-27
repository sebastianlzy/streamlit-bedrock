# do_vs_dont = "Recommend some creative space to visit in Sydney. Don't include gallery or museums."
do_vs_dont = "Recommend some creative space to visit in Sydney. Exclude gallery and museums."

zero_shot_prompt = """
A spider 
"""

few_shot_prompt = """
A dog has 4 legs.
A sheep has 4 legs.
A spider
"""

chain_of_thoughts_prompt = """
This is awesome! // Positive => Negative
This is bad! // Negative => Positive
Wow that movie was rad! // Positive => Negative
What a horrible show! //
"""

simple_prompt = "Explain what general relativity is to an 8 year old."

meeting_transcribe_prompt = """
Meeting transcript: 
Miguel: Hi Brant, I want to discuss the workstream  for our new product launch 
Brant: Sure Miguel, is there anything in particular you want to discuss? 
Miguel: Yes, I want to talk about how users enter into the product. 
Brant: Ok, in that case let me add in Namita. 
Namita: Hey everyone 
Brant: Hi Namita, Miguel wants to discuss how users enter into the product. 
Miguel: its too complicated and we should remove friction.  for example, why do I need to fill out additional forms?  I also find it difficult to find where to access the product when I first land on the landing page. 
Brant: I would also add that I think there are too many steps. 
Namita: Ok, I can work on the landing page to make the product more discoverable but brant can you work on the additonal forms? 
Brant: Yes but I would need to work with James from another team as he needs to unblock the sign up workflow.  Miguel can you document any other concerns so that I can discuss with James only once? 
Miguel: Sure. 
From the meeting transcript above, Create a list of action items for each person. 
"""

article_summarisation_prompt = """
Meet Carbon Maps, a new French startup that raised $4.3 million (€4 million) just a few weeks after its inception. The company is building a software-as-a-service platform for the food industry so that they can track the environmental impact of each of their products in their lineup. The platform can be used as a basis for eco ratings. 

While there are quite a few carbon accounting startups like Greenly, Sweep, Persefoni and Watershed, Carbon Maps isn’t an exact competitor as it doesn’t calculate a company’s carbon emissions as a whole. It doesn’t focus on carbon emissions exclusively either. Carbon Maps focuses on the food industry and evaluates the environmental impact of products — not companies. 

Co-founded by Patrick Asdaghi, Jérémie Wainstain and Estelle Huynh, the company managed to raise a seed round with Breega and Samaipata — these two VC firms already invested in Asdaghi’s previous startup, FoodChéri. 

FoodChéri is a full-stack food delivery company that designs its own meals and sells them directly to end customers with an important focus on healthy food. It also operates Seazon, a sister company for batch deliveries. The startup was acquired by Sodexo a few years ago. 

“On the day that I left, I started working on food and health projects again,” Asdaghi told me. “I wanted to make an impact, so I started moving up the supply chain and looking at agriculture.” 

And the good news is that Asdaghi isn’t the only one looking at the supply chain of the food industry. In France, some companies started working on an eco-score with a public agency (ADEME) overseeing the project. It’s a life cycle assessment that leads to a letter rating from A to E. 

While very few brands put these letters on their labels, chances are companies that have good ratings will use the eco-score as a selling point in the coming years. 

But these ratings could become even more widespread as regulation is still evolving. The European Union is even working on a standard — the Product Environmental Footprint (PEF). European countries can then create their own scoring systems based on these European criteria, meaning that food companies will need good data on their supply chains. 

“The key element in the new eco-score that’s coming up is that there will be some differences within a product category because ingredients and farming methods are different,” Asdaghi said. “It’s going to take into consideration the carbon impact, but also biodiversity, water consumption and animal welfare.” 

For instance, when you look at ground beef, it’s extremely important to know whether farmers are using soy from Brazil or grass to feed cattle. 

“We don’t want to create the ratings. We want to create the tools that help with calculations — a sort of SAP,” Asdaghi said. 

So far, Carbon Maps is working with two companies on pilot programs as it’s going to require a ton of work to cover each vertical in the food industry. The startup creates models with as many criteria as possible to calculate the impact of each criteria. It uses data from standardized sources like GHG Protocol, IPCC, ISO 14040 and 14044. 

The company targets food brands because they design the recipes and select their suppliers. Eventually, Carbon Maps hopes that everybody across the supply chain is going to use its platform in one way or another. 

“You can’t have a true climate strategy if you don’t have some collaboration across the chain,” Asdaghi said. 

## 

Summarize the above text in 5 bullets.
"""

content_generation_prompt = """
I'd like you to rewrite the following paragraph using the following instructions: "understandable to a 5th grader". 
"In 1758, the Swedish botanist and zoologist Carl Linnaeus published in his Systema Naturae, the two-word naming of species (binomial nomenclature). Canis is the Latin word meaning "dog", and under this genus, he listed the domestic dog, the wolf, and the golden jackal. "
Please put your rewrite in <rewrite></rewrite> tags.
"""

create_a_table_of_product_description_prompt = """
Product: Sunglasses. 
Keywords: polarized, designer, comfortable, UV protection, aviators. 

Create a table that contains five variations of a detailed product description for the product listed above, each variation of the product description must use all the keywords listed.

"""

extract_topics_and_sentiment_from_reviews = """

Review: 
Extremely old cabinets, phone was half broken and full of dust. Bathroom door was broken, bathroom floor was dirty and yellow. Bathroom tiles were falling off. Asked to change my room and the next room was in the same conditions. The most out of date and least maintained hotel i ever been on.
Extracted sentiment:
{“Cleaning”: “Negative”, “Hotel Facilities”: “Negative”, “Room Quality”: “Negative”}
## 
Review: 
Great experience for two teenagers. We would book again. Location good. 
Extracted sentiment:
{“Location”: “Positive”}
## 
Review: 
Pool area was definitely a little run down and did not look like the pics online at all. Bathroom in the double room was kind of dumpy.
Extracted sentiment:
{“Pool”: “Negative”, “Room Quality”: “Negative”}
## 
Review: 
Roof top’s view is gorgeous and the lounge area is comfortable. The staff is very courteous and the location is great. The hotel is outdated and the shower need to be clean better. The air condition runs all the time and cannot be control by the temperature control setting. 
Extracted sentiment:
{“Cleaning”: “Negative”, “AC”: “Negative”, “Room Quality”: “Negative”, “Service”: “Positive”, “View”: “Positive”, “Hotel Facilities”: “Positive”}
## 
Review: 
First I was placed near the elevator where it was noises, the TV is not updated, the toilet was coming on and off. There was no temp control and my shower curtain smelled moldy. Not sure what happened to this place but only thing was a great location.
Extracted sentiment:
{“Cleaning”: “Negative”, “AC”: “Negative”, “Room Quality”: “Negative”, “Location”: “Positive”}
## 
Review: 
This is a very well located hotel and it’s nice and clean. Would stay here again. 
Extracted sentiment:
{“Cleaning”: “Positive”, “Location”: “Positive”}
## 
Review: 
Hotel is horrendous. The room was dark and dirty. No toilet paper in the bathroom. Came here on a work trip and had zero access to WiFi even though their hotel claims to have WiFi service. I will NEVER return.
Extracted sentiment:
{“Cleaning”: “Negative”, “WiFi”: “Negative”, “Room Quality”: “Negative”, “Service”: “Negative”}
## 
Review: 
The rooms are small but clean and comfortable. The front desk was really busy and the lines for check-in were very long but the staff handled each person in a professional and very pleasant way. We will stay there again. 
Extracted sentiment:
{“Cleaning”: “Positive”, “Service”: “Positive”}
## 
Review: 
The stay was very nice would stay again. The pool closes at 7 pm and doesn’t open till 11am m. That sucked. Also our wifi went out the entire last day we were there. Thats sucked too. Overall was a nice enough stay and I love the location.
Extracted sentiment:

"""

generate_product_descriptions_prompt = """

Write an engaging product description for a clothing eCommerce site. Make sure to include the following features in the description. 
Product: Humor Men's Graphic T-Shirt. 
Features: 
- Soft cotton 
- Short sleeve
- Have a print of Einstein's quote: "artificial intelligence is no match for natural stupidity” 
Description: 

"""

information_extraction_prompt = """

Please precisely copy any email addresses from the following text and then write them, one per line. Only write an email address if it's precisely spelled out in the input text. If there are no email addresses in the text, write "N/A". Do not say anything else. 
"Phone Directory:
John Latrabe, 800-232-1995, john909709@geemail.com
Josie Lana, 800-759-2905, josie@josielananier.com
Keven Stevens, 800-980-7000, drkevin22@geemail.com 
Phone directory will be kept up to date by the HR manager." 

"""

multiple_choice_classification = """

You are a customer service agent that is classifying emails by type. I want you to give your answer and then explain it. 
How would you categorize this email? 
<email>
Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?
</email> 

Categories are: 
(A) Pre-sale question 
(B) Broken or defective item 
(C) Billing question 
(D) Other (please explain)

"""

outline_generation_prompt = """

Suggest an outline for a blog post based on a title. 
Title: How I put the pro in prompt engineering 


"""

question_and_answer_prompt = """

Have you heard of the following company? 
"Anthropic" 
Please reply with Yes or No. Do not say anything else.

"""

remove_pii_prompt = """

Here is some text. We want to remove all personally identifying information from this text and replace it with XXX. It's very important that names, phone numbers, and email addresses, gets replaced with XXX. 
Here is the text, inside <text></text> XML tags
<text>
   Joe: Hi Hannah!
   Hannah: Hi Joe! Are you coming over?  
   Joe: Yup! Hey I, uh, forgot where you live." 
   Hannah: No problem! It's 4085 Paco Ln, Los Altos CA 94306.
   Joe: Got it, thanks!  
</text> 
Please put your sanitized version of the text with PII removed in <response></response> XML tags 

"""

summarise_the_key_takeaways = """
The seeds of a machine learning (ML) paradigm shift have existed for decades, but with the ready availability of scalable compute capacity, a massive proliferation of data, and the rapid advancement of ML technologies, customers across industries are transforming their businesses. Just recently, generative AI applications like ChatGPT have captured widespread attention and imagination. We are truly at an exciting inflection point in the widespread adoption of ML, and we believe most customer experiences and applications will be reinvented with generative AI. AI and ML have been a focus for Amazon for over 20 years, and many of the capabilities customers use with Amazon are driven by ML. Our e-commerce recommendations engine is driven by ML; the paths that optimize robotic picking routes in our fulfillment centers are driven by ML; and our supply chain, forecasting, and capacity planning are informed by ML. Prime Air (our drones) and the computer vision technology in Amazon Go (our physical retail experience that lets consumers select items off a shelf and leave the store without having to formally check out) use deep learning. Alexa, powered by more than 30 different ML systems, helps customers billions of times each week to manage smart homes, shop, get information and entertainment, and more. We have thousands of engineers at Amazon committed to ML, and it’s a big part of our heritage, current ethos, and future. At AWS, we have played a key role in democratizing ML and making it accessible to anyone who wants to use it, including more than 100,000 customers of all sizes and industries. AWS has the broadest and deepest portfolio of AI and ML services at all three layers of the stack. We’ve invested and innovated to offer the most performant, scalable infrastructure for cost-effective ML training and inference; developed Amazon SageMaker, which is the easiest way for all developers to build, train, and deploy models; and launched a wide range of services that allow customers to add AI capabilities like image recognition, forecasting, and intelligent search to applications with a simple API call. This is why customers like Intuit, Thomson Reuters, AstraZeneca, Ferrari, Bundesliga, 3M, and BMW, as well as thousands of startups and government agencies around the world, are transforming themselves, their industries, and their missions with ML. We take the same democratizing approach to generative AI: we work to take these technologies out of the realm of research and experiments and extend their availability far beyond a handful of startups and large, well-funded tech companies. That’s why today I’m excited to announce several new innovations that will make it easy and practical for our customers to use generative AI in their businesses. Summarize the above text into three key takeaways. 
"""

write_an_article = """

Write an informational article for children about how birds fly.  Compare how birds fly to how airplanes fly.  Make sure to use the word "Thrust" at least three times.
"""

write_a_promo_doc = """
Bob has been at the company for 3.5 years and is up for promotion for senior manager.  
Bob has the following accomplishments: 
- led the delivery of a new software product for productivity. 
- mentored 10 employees. 
- presented to 20 different end customers  which converted into 18 new users. 

 
 Using  the above content, write a business professional narrrative about Bobs performance.  The narrative must include the reasons why bob should be promoted and mention three amazons leadership principles. 

"""

code_generation_prompt = """
Write the Python code to plot the cosine curve.

"""

custom_prompt = f"""
Explain what general relativity is to an 8 year old.
"""
