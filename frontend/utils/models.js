import { get_emnist_images, get_graffiti_images, get_image, get_mnist_images, get_met_images } from "./generate_images"

export const models = [
  { id: 1, name: "base", func: (params) => get_image(params) },
  { id: 2, name: "EMNIST", func: (params) => get_emnist_images(params) },
  { id: 3, name: "MNIST", func: (params) => get_mnist_images(params) },
  { id: 4, name: "MetFaces", func: (params) => get_met_images(params) },
  { id: 5, name: "Graffiti", func: (params) => get_graffiti_images(params) },
  // { id: 6, name: "Jujutsu Kaisen" },
]
