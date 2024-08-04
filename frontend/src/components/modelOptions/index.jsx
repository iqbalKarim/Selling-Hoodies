import { useContext } from "react"
import styles from "./modelOptions.module.css"
import { models } from "../../../utils/models"
import { ActiveModelContext } from "../../context/activeModelContext"

export default function ModelOptions({}) {
  const { activeModel, setActiveModel } = useContext(ActiveModelContext)

  function changeModel(model) {
    setActiveModel(model)
  }

  return (
    <div style={{ borderRight: "1px solid #555", paddingRight: 20, marginRight: 20, display: "flex" }}>
      <p style={{ marginLeft: 20 }}>
        {models.map((model) => (
          <button
            key={model.id}
            className={activeModel.id == model.id ? styles.button_inactive : styles.button}
            onClick={() => changeModel(model)}
          >
            {model.name}
          </button>
        ))}
      </p>
    </div>
  )
}
