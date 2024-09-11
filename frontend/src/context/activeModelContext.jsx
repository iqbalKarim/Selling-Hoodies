import { createContext, useState } from "react"
import { models } from "../../utils/models"

export const ActiveModelContext = createContext({ activeModel: null, setActiveModel: () => null })

const ActiveModelContextProvider = ({ children }) => {
  const [activeModel, setActiveModel] = useState(models[0])

  const value = { activeModel, setActiveModel }

  return <ActiveModelContext.Provider value={value}>{children}</ActiveModelContext.Provider>
}

export default ActiveModelContextProvider
