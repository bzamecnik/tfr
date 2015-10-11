ffmpeg -i piano-tsne.mp4 -i /Users/bzamecnik/Documents/harmoneye-labs/harmoneye/data/wav/c-scale-piano-mono.mp3 -map 0 -map 1 -codec copy -shortest piano-tsne-with-audio.mp4
