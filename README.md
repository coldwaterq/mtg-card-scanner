### Setup
1. install podman and podman-compose
2. Install Milvus
    - `podman compose up -d`
    - docker-compose.yml in this is downloaded from https://github.com/milvus-io/milvus/releases/
3. python dbcreate.py
    - this downloads the card images into onedrive, calculates the embedding, and adds that to the db
    - line 95 to 100 are commented out and can be used to only download a specific set, card name, or image type

### Run
1. open podman and start the mtg-card-scanner pod
2. run `python scan.py CSVFILENAME`
3. This will open a window showing your web cam.
4. If something is roughly card shaped it will zoom in on it, and run it through the DB
5. if there is a close enough match, the image will freeze with the card information overlayed on it
    - In this state you can either:
        - press escape to tell the camera to look again
        - press enter to add the standard card info to your csv along with price
        - press tab to add the foil card info to your csv along with price
        - press an ascii character (letter, number, symbol) to refine the search to both cards with a name starting with that string, and card set numbers (SET-123) starting with that string.
    - If keyboard input isn't be registered, click in the image window and try again
6. while not showing card information to gracefully shutdown the program

