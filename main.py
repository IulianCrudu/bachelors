from typing import Dict, List, Tuple, Union
from uuid import UUID
from torch import Tensor

from vehicle_tracking import vehicle_tracking
from optical_flow import dense_optical_flow

from utils import Car, Coordinate


def corners_from_coordinate(coordinate_list: Union[List[float], Tensor]) -> Tuple[Coordinate, Coordinate]:
    top_left = Coordinate(x=int(coordinate_list[0]), y=int(coordinate_list[1]))
    bottom_right = Coordinate(x=int(coordinate_list[2]), y=int(coordinate_list[3]))
    return top_left, bottom_right


# TODO: how do I compute the cars lost from frame1 to frame2?


if __name__ == "__main__":
    source = "data/test/"
    frames = vehicle_tracking(source, save_source="runs/detect/test")

    prev_frame = frames[0]
    current_frame = frames[1]
    cars: Dict[UUID, Car] = {}
    removed_cars = 0

    for coordinate in current_frame.coordinates:
        top_left_corner, bottom_right_corner = corners_from_coordinate(coordinate)
        car = Car(
            top_left_corner=top_left_corner,
            bottom_right_corner=bottom_right_corner
        )
        cars[car.id] = car
    optical_flow = dense_optical_flow(path1=prev_frame.path, path2=current_frame.path)
    current_cars = list(cars.values())

    for frame in frames[2:]:
        # Predict the location of each car in the next frame
        for car in current_cars:
            top_left_vector = optical_flow[car.top_left_corner.y][car.top_left_corner.x]
            bottom_right_vector = optical_flow[car.bottom_right_corner.y][car.bottom_right_corner.x]

            predicted_top_left = Coordinate(
                x=car.top_left_corner.x + top_left_vector[0],
                y=car.top_left_corner.y + top_left_vector[1],
            )
            predicted_bottom_right = Coordinate(
                x=car.bottom_right_corner.x + bottom_right_vector[0],
                y=car.bottom_right_corner.y + bottom_right_vector[1],
            )

            # Try to match the predictions to one of the cars in the frame

            found = False
            count = 0
            for coordinate in frame.coordinates:
                # TODO:
                #  avoid using the same coordinate for 2+ cars
                #       this also means that the new cars from the frame are not added
                #       have all the frames in a set, once used pop it out from the set, at the end of this loop all the coordinates that are left are new cars
                #  if multiple coordinates match the same car, pick the closest one

                top_left_corner, bottom_right_corner = corners_from_coordinate(coordinate)

                if (
                    predicted_top_left.is_equal_with_error(top_left_corner)
                    and predicted_bottom_right.is_equal_with_error(bottom_right_corner)
                ):
                    count += 1
                    if count == 1:
                        car.top_left_corner = top_left_corner
                        car.bottom_right_corner = bottom_right_corner
                        found = True
                    # break

            if count > 1:
                print(f"Found multiple coordinates for car {car}.")

            if not found:
                print(f"Car {car} not found.")
                del cars[car.id]
                removed_cars += 1

        optical_flow = dense_optical_flow(path1=current_frame.path, path2=frame.path)
        current_frame = frame
        current_cars = list(cars.values())

    """
        For each found car, associate an ID.
        
        Apply optical flow between 2 consecutive frames.
        In the 3rd frame, check whether the car exists on the expected location(by using the vectors from the optical flow)
        If it exists, we know it's the same car.
    """
    # for image in images[1:]:
    #     prev_path, prev_coordinates = prev_image.path, prev_image.coordinates
    #     path, coordinates = image.path, image.coordinates
    #
    #     # Compute
    #     dense_optical_flow(prev_path, path)
    #
    #     prev_image = image
    #

    print(f"Cars counted: {len(cars) + removed_cars}")
