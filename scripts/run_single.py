from video_understanding import VideoUnderstandingSystem

Vus = VideoUnderstandingSystem(
    video_duration = 6020.5,
    question = '''How is the situation in the first half of the match?
(A) In the first half, both Arsenal and Liverpool had opportunities to score but failed to capitalize. Arsenal created several chances, including one shot that narrowly missed the goal frame, but were unable to convert. Liverpool, on the other hand, managed to score twice. Despite Arsenal's efforts to find the back of the net, Liverpool's goalkeeper made some impressive saves to maintain their lead. At halftime, the score stood at 2-0 in favor of Liverpool
(B) In the first half, both Arsenal and Liverpool had opportunities to score but failed to capitalize. Arsenal created several chances, including one shot that narrowly missed the goal frame, but were unable to convert. Liverpool also had a few opportunities, but Arsenal's goalkeeper made some impressive saves to keep them out. Despite both teams' efforts to score, the first half ended with the score still 0-0
(C) In the first half, both Arsenal and Liverpool had opportunities to score and both managed to find the back of the net once. Arsenal created several chances, including one shot that narrowly missed the goal frame, but they eventually scored. Similarly, Liverpool also capitalized on their chances and scored one goal. Despite both teams' efforts to add to their tally, the score remained 1-1 at halftime
(D) In the first half, both Arsenal and Liverpool had opportunities to score but failed to capitalize. Arsenal created several chances, including one shot Liverpool also had a few opportunities, but Arsenal's goalkeeper made some impressive saves to keep them out. The first half ended with the score still 1-0''',
    frame_path = '/video_database/rSE2YPcv89U/frames',
    sub_path = '/subtitles/rSE2YPcv89U.json',
    log_path = '/logs/debug_v2',
    data_name = "lv_bench"
)
final_result = Vus.run()