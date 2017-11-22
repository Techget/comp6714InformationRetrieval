This project tries to make the similar adjectives have very small cosDistance.

To see the detailed requirement see: proj2_spec.ipynb
To run the program on the local, run `niubi.py`, it will reads from BBC_Data.zip, and train the word2vec, then will write to 
adjective_embedding.txt, save the trained model, then it will compute the top_100 nearest word for each test word, and compare with
the dev_set groud truth, to compute the score of the trained model

To check a clean part, please have a look under `submission/`, also the report.pdf in there articulates all the aspects of this project.

