# visual_reco

**1. emotions_detection**

--input : video.mp4

--output : json{
    'mimika_timestamp' : timestamp[s],
    'emotions' : {timestamp[s] : 'emotion'}
}

mimika_timestamp <- timestamp w sekundach znalezienia różnicy w emocjach na twarzy, jeśli None - nie znaleziono


emotions <- dla każdej sekundy nagrania została podana najmocniej wykryta emocja


**2. background_person_detection**

--input : video.mp4

--output : json{
    'background_person' : [timestamp[s]]
}

background_person <- lista timestampów[s] znalezienia drugiej osoby w tle