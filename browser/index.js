import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

// Reference the elements that we will need
const status = document.getElementById('status');
const fileUpload = document.getElementById('file-upload');
const imageContainer = document.getElementById('image-container');

function cosineSimilarity(query_embeds, database_embeds) {
    const EMBED_DIM = 512;

    const numDB = database_embeds.length;// / EMBED_DIM;
    const similarityScores = new Array(numDB);

    for (let i = 0; i < numDB; ++i) {
        // const startOffset = i * EMBED_DIM;
        const dbVector = database_embeds[i];//.slice(startOffset, startOffset + EMBED_DIM);
        let dotProduct = 0;
        let normEmbeds = 0;
        let normDB = 0;

        for (let j = 0; j < EMBED_DIM; ++j) {
            const embedValue = query_embeds[j];
            const dbValue = dbVector[j];

            dotProduct += embedValue * dbValue;
            normEmbeds += embedValue * embedValue;
            normDB += dbValue * dbValue;
        }

        similarityScores[i] = dotProduct / (Math.sqrt(normEmbeds) * Math.sqrt(normDB));
    }

    return similarityScores;
}

// getCachedFile is not created, and uses nodejs stuff.
// get the mebeddings
var embeddings
var embeddingKeys
await fetch("embeddings/mh3.json")
    .then( response => response.json())
    .then( async data => {
        embeddingKeys = Object.keys(data)
        embeddings = []
        for (var k of embeddingKeys){
            embeddings.push(new Float32Array(data[k]))
        }
});
console.log(embeddings)
alert();

// Create a new object detection pipeline
status.textContent = 'Loading model...';
const features =  await pipeline('image-feature-extraction', 'Xenova/clip-vit-base-patch32');
status.textContent = 'Ready';

fileUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();

    // Set up a callback when the file is loaded
    reader.onload = function (e2) {
        imageContainer.innerHTML = '';
        const image = document.createElement('img');
        image.src = e2.target.result;
        imageContainer.appendChild(image);
        detect(image);
    };
    reader.readAsDataURL(file);
});


// Detect objects in the image
async function detect(img) {
    status.textContent = 'Analysing...';
    const output = await features(img.src);
    status.textContent = output;
    var scores = cosineSimilarity(output.data,embeddings)
    var best = ""
    var bestScore = 0
    for (var i = 0; i< scores.length; i++){
        if( scores[i] > bestScore){
            bestScore = scores[i];
            best = embeddingKeys[i]
        }
    }
    console.log(embeddingKeys.indexOf("mh3-16"))
    console.log(scores[embeddingKeys.indexOf("mh3-16")])
    console.log(bestScore, best);
}

// Render a bounding box and label on the image
function renderBox({ box, label }) {
    const { xmax, xmin, ymax, ymin } = box;

    // Generate a random color for the box
    const color = '#' + Math.floor(Math.random() * 0xFFFFFF).toString(16).padStart(6, 0);

    // Draw the box
    const boxElement = document.createElement('div');
    boxElement.className = 'bounding-box';
    Object.assign(boxElement.style, {
        borderColor: color,
        left: 100 * xmin + '%',
        top: 100 * ymin + '%',
        width: 100 * (xmax - xmin) + '%',
        height: 100 * (ymax - ymin) + '%',
    })

    // Draw label
    const labelElement = document.createElement('span');
    labelElement.textContent = label;
    labelElement.className = 'bounding-box-label';
    labelElement.style.backgroundColor = color;

    boxElement.appendChild(labelElement);
    imageContainer.appendChild(boxElement);
}