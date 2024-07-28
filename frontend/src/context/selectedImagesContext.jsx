import { createContext, useState } from "react"

export const SelectedImagesContext = createContext(null)

export const SelectedImagesContextProvider = ({ children }) => {
  const [selectedImages, setSelectedImages] = useState([])

  const values = { selectedImages, setSelectedImages }

  return (
    <>
      <SelectedImagesContext.Provider value={values}>{children}</SelectedImagesContext.Provider>
    </>
  )
}
