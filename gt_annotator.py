
import sys
import pygame
import traceback
import json

from AccessMath.data.meta_data_DB import MetaDataDB
from AM_CommonTools.configuration.configuration import Configuration

from AccessMath.annotation.gt_content_annotator import GTContentAnnotator


def get_video_files(argvs):
    # read the config ...
    config = Configuration.from_file(argvs[1])

    output_dir = config.get_str("OUTPUT_PATH")
    database_filename = config.get_str("VIDEO_DATABASE_PATH")
    try:
        database = MetaDataDB.from_file(database_filename)
    except:
        print("Invalid database file")
        return None, None, None

    current_lecture = None
    if len(argvs) >= 3:
        # now search for specified lecture
        lecture_name = argvs[2].lower()

        for lecture in database.lectures:
            if lecture.title.lower() == lecture_name:
                current_lecture = lecture
                break
    else:
        lecture_name = None

    if current_lecture is None:
        print("Lecture not found in database")
        print("Available lectures:")

        candidate_lectures = database.get_lectures(lecture_name, True)

        tempo_str = ""
        for idx, lecture in enumerate(candidate_lectures):
            tempo_str += lecture.title + ("\t" if (idx + 1) % 4 > 0 else "\n")
        print(tempo_str)

        return None, None, None

    video_based_annotation = config.get_bool("VIDEO_BASED_ANNOTATIONS", True)
    if video_based_annotation:
        main_video_path = config.get_str("VIDEO_FILES_PATH")
    else:
        main_video_path = config.get_str("OUTPUT_FRAME_EXPORT")

    for video in current_lecture.main_videos:
        if video["type"].lower() == "video" and not video_based_annotation:
            print("Found video on Database but configuration does not uses video based annotations")
            print("- Trying to read an export")

            # override
            video["type"] = "ImageList"
            video["path"] = main_video_path + "/" + current_lecture.title + "/JPEGImages"
            video["format"] = config.get("OUTPUT_FRAME_EXPORT_FORMAT")
        else:
            video["path"] = main_video_path + "/" + video["path"]

    return output_dir, database, current_lecture

def main():
    if len(sys.argv) < 2:
        print("Usage: python gt_annotator.py config lecture [metrics]")
        print("Where")
        print("\tconfig\t\t= Configuration file")
        print("\tlecture\t\t= Lecture video to process")
        print("\tmetrics\t\t= Optional, use pre-computed video metrics")
        print("")
        return

    output_dir, database, current_lecture = get_video_files(sys.argv)
    if current_lecture is None or current_lecture.main_videos is None:
        return

    if len(sys.argv) >= 4:
        metrics_filename = sys.argv[3]
        with open(metrics_filename, "r") as in_file:
            video_metrics = json.load(in_file)

        if current_lecture.title in video_metrics:
            lecture_metrics = video_metrics[current_lecture.title]
        else:
            print("The video metrics file does not contain information about current lecture")
            return
    else:
        lecture_metrics = None

    output_prefix = output_dir + "/" + database.output_annotations + "/" + database.name + "_" + current_lecture.title.lower()
    print("Annotation Prefix: " + output_prefix)

    pygame.init()
    pygame.display.set_caption('Access Math - Ground Truth Annotation Tool - ' + database.name + "/" + current_lecture.title)
    screen_w = 1500
    screen_h = 900
    window = pygame.display.set_mode((screen_w, screen_h))
    background = pygame.Surface(window.get_size())
    background = background.convert()

    if "forced_width" in current_lecture.parameters:
        forced_res = (current_lecture.parameters["forced_width"], current_lecture.parameters["forced_height"])
        print("Video Resolution will be forced to : " + str(forced_res))
    else:
        forced_res = None

    main_menu = GTContentAnnotator(window.get_size(), current_lecture.main_videos, database.name, current_lecture.title,
                                   output_prefix, forced_res)

    if lecture_metrics is not None:
        main_menu.player.video_player.update_video_metrics(lecture_metrics)

    current_screen = main_menu
    current_screen.prepare_screen()
    prev_screen = None

    while not current_screen is None:
        #detect when the screen changes...
        if current_screen != prev_screen:
            #remember last screen...
            prev_screen = current_screen

        #capture events...
        current_events = pygame.event.get()
        try:
            current_screen = current_screen.handle_events(current_events)
        except Exception as e:
            print("An exception ocurred")
            print(e)
            traceback.print_exc()

        if current_screen != prev_screen:
            if current_screen != None:
                #prepare the screen for new display ...
                current_screen.prepare_screen()

        #draw....
        background.fill((0, 0, 0))

        if not current_screen is None:
            current_screen.render(background)

        window.blit(background, (0, 0))
        pygame.display.flip()


if __name__ == "__main__":
    main()