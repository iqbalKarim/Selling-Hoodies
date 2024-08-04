import { useState } from "react"
import styles from "./modelOptions.module.css"

export default function ModelOptions(){
    const [activeModel, setActiveModel] = useState(1)

    const models = [
        {id:1, name: 'base'},
        {id:2, name: 'EMNIST'},
        {id:3, name: 'MNIST'},
        {id:4, name: 'MetFaces'},
        {id:5, name: 'Graffiti'},
        {id:6, name: 'Jujutsu Kaisen'}
    ]

    function changeModel(model){
        setActiveModel(model.id)
    }
    
    return (
        <div style={{borderRight: '1px solid #555', paddingRight: 20, marginRight: 20, display: 'flex'}}>
            <p style={{marginLeft: 20}}>
                {models.map(model => 
                    <button 
                        key={model.id} 
                        className={activeModel == model.id ? styles.button_inactive : styles.button}
                        onClick={() => changeModel(model)} 
                    >  
                        {model.name}
                    </button>
                )}
            </p>
        </div>
    )
}