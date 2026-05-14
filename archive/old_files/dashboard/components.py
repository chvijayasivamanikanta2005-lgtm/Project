import streamlit as st
import streamlit.components.v1 as components

def render_led_switch(checked, key):
    led_switch = components.declare_component("led_switch", path="assets/led_switch")
    return led_switch(
        checked=checked,
        key=key,
        default=checked,
    )
