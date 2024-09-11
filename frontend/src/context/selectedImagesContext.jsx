import { createContext, useState } from "react"

export const SelectedImagesContext = createContext({
  selectedImages: null,
  setSelectedImages: () => null,
  activeImage: null,
  setActiveImage: () => null,
  hoodieBackground: null,
  setHoodieBackground: () => null,
})

export const SelectedImagesContextProvider = ({ children }) => {
  const [selectedImages, setSelectedImages] = useState([])
  const [activeImage, setActiveImage] = useState(null)
  const [hoodieBackground, setHoodieBackground] = useState("black")

  const values = { selectedImages, setSelectedImages, activeImage, setActiveImage, hoodieBackground, setHoodieBackground }

  return (
    <>
      <SelectedImagesContext.Provider value={values}>{children}</SelectedImagesContext.Provider>
    </>
  )
}
